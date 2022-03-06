import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss

class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None,
                hidden_states2=None, attention_mask2=None,
                l=None, mix_layer=1000, head_mask=None):
        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None

        # Perform mix till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i <= mix_layer:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            elif i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1 - l) * hidden_states2
                    attention_mask = attention_mask.long() | attention_mask2.long()
                    ## sentMix: (bsz, len, hid)
                    # hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1-l)*hidden_states2[:, 0, :]
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs


class BertModel4Mix(BertPreTrainedModel, nn.Module):

    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    # def _resize_token_embeddings(self, new_num_tokens):
    #     old_embeddings = self.embeddings.word_embeddings
    #     new_embeddings = self._get_resized_embeddings(
    #         old_embeddings, new_num_tokens)
    #     self.embeddings.word_embeddings = new_embeddings
    #     return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000, head_mask=None, inputs_embeds = None):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids,inputs_embeds = inputs_embeds)

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0
            embedding_output2 = self.embeddings(input_ids2, token_type_ids=token_type_ids2)
            encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                                           embedding_output2, extended_attention_mask2,
                                           l, mix_layer, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, embedding_output) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class MixText(BertPreTrainedModel, nn.Module):
    def __init__(self, config):
        super(MixText, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel4Mix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000, inputs_embeds = None, labels = None):

        if input_ids2 is not None:
            outputs = self.bert(input_ids, attention_mask, token_type_ids,
                                input_ids2, attention_mask2, token_type_ids2,
                                l, mix_layer, inputs_embeds = inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # return logits, outputs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )





