import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import os
from tqdm import tqdm

import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import MyTextFoolerJin2019

class BertSModel():
    '''
    bert for sentence classification
    S: refer to sentence
    '''
    def __init__(self,model_type = 'bert-base-uncased',fine_tune_dir = None, 
                       output_dir = '/mnt/cloud/bairu/repos/TextAttack/checkpoints/bert-sst-at/',
                       cache_dir = '/mnt/cloud/bairu/model_cache/bert_model/bert-base-uncased/',
                       dataset_dir = '/mnt/cloud/bairu/repos/text_grad/sst-2/',
                       max_len = 100,
                       device = torch.device("cuda"),
                       num_labels = 2):
        if fine_tune_dir != None:
            self.tokenizer = BertTokenizer.from_pretrained(fine_tune_dir)
            self.model = BertForSequenceClassification.from_pretrained(fine_tune_dir, num_labels = num_labels)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_type, cache_dir = cache_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_type,cache_dir = cache_dir, num_labels = num_labels)
        self.device = device
        self.model = self.model.to(self.device)
        # self.model.cuda()
        self.max_len = max_len
        self.optimizer = AdamW(self.model.parameters(), lr = 2e-5)
        self.max_batch_size = 128
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
    
    def load_dataset(self,):
        train_corpus, train_label = self.process_dataset(split = 'train')
        valid_corpus, valid_label = self.process_dataset(split = 'valid')
        test_corpus, test_label = self.process_dataset(split = 'test')
        return train_corpus, train_label,valid_corpus,valid_label,test_corpus, test_label

    def process_dataset(self, split):
        sen_list = []
        label_list = []
        with open(self.dataset_dir + split + '.tsv','r',encoding='utf-8') as f:
            for line in f:
                sentence, label = line.split('\t',1)
                sen_list.append(sentence.strip())
                label_list.append(int(label.strip()))
        return sen_list,label_list

    def tokenize_corpus(self,corpus):
        tokenized_list = []
        attention_masks = []
        sample_sentence = []
        for i in range(len(corpus)):
            sentence = corpus[i][:]
            result = self.tokenizer.encode_plus(sentence,max_length = self.max_len,padding = "max_length",return_attention_mask = True, truncation = True, add_special_tokens = True)
            sentence_ids = result['input_ids']
            mask = result['attention_mask']
            attention_masks.append(mask)
            tokenized_list.append(sentence_ids)
            # tokenized_list.append(tokenizer.encode(sentence,max_length = max_len,pad_to_max_length = True))
        # print(sample_sentence)
        return np.array(tokenized_list),np.array(attention_masks)

    def fine_tune(self,batch_size = 32):
        # self.model.save_pretrained(self.output_dir)
        # self.tokenizer.save_pretrained(self.output_dir)
        self.model.train()
        train_corpus, train_label,valid_corpus, valid_label,test_corpus, test_label = self.load_dataset()
        train_xs, train_masks = self.tokenize_corpus(train_corpus)
        train_ys = np.array(train_label)
        valid_xs, valid_masks = self.tokenize_corpus(valid_corpus)
        valid_ys = np.array(valid_label)
        test_xs, test_masks = self.tokenize_corpus(test_corpus)
        test_ys = np.array(test_label)
        batches_per_epoch = train_xs.shape[0]//batch_size
        global_acc = 0
        # output_dir = 'albert_sst_xxl'
        for i in range(3):
            selection = np.random.choice(train_xs.shape[0],size = train_xs.shape[0],replace = False)
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for idx in range(batches_per_epoch):
                batch_idx = np.array(selection[idx*batch_size:(idx+1)*batch_size])
                batch_xs = torch.LongTensor(train_xs[batch_idx]).to(self.device)
                batch_ys = torch.LongTensor(train_ys[batch_idx]).to(self.device)
                batch_masks = torch.LongTensor(train_masks[batch_idx]).to(self.device)
                # print(batch_xs.size())
                self.optimizer.zero_grad()
                # model.zero_grad()
                result = self.model(input_ids = batch_xs,labels = batch_ys,attention_mask = batch_masks)
                loss = result.loss
                logits = result.logits
                epoch_loss += loss.item()
                epoch_accuracy += torch.argmax(logits,dim = 1).eq(batch_ys).sum().item()/batch_size
                loss.backward()
                self.optimizer.step()
            epoch_loss /= batches_per_epoch
            epoch_accuracy /= batches_per_epoch
            print(i,' ',epoch_loss, ' ',epoch_accuracy)    
            print('Train accuracy = ', self.evaluate_accuracy(train_xs,train_ys,train_masks,batch_size))
            local_acc = self.evaluate_accuracy(valid_xs, valid_ys, valid_masks, batch_size)
            print("valid accuracy = ", local_acc)
            if local_acc > global_acc:
                global_acc = local_acc
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)

        self.model = BertForSequenceClassification.from_pretrained(self.output_dir)
        self.model.to(self.device)
        print("Test accuracy = ", self.evaluate_accuracy(test_xs,test_ys,test_masks,batch_size))
        print("All done")      
        self.model.eval() 

    def eval_on_test(self):
        self.model.eval()
        train_corpus, train_label,valid_corpus, valid_label,test_corpus, test_label = self.load_dataset()
        test_xs, test_masks = self.tokenize_corpus(test_corpus)
        test_ys = np.array(test_label)
        print("Test accuracy = ", self.evaluate_accuracy(test_xs,test_ys,test_masks,batch_size = 8))

    def evaluate_accuracy(self,test_xs,test_ys,attention_xs, batch_size):
        test_batches = test_xs.shape[0]//batch_size
        test_accuracy = 0.0
        self.model.eval()
        for i in range(test_batches):
            test_idx = range(i * batch_size,(i + 1)*batch_size)
            xs = torch.LongTensor(test_xs[test_idx,:]).to(self.device)
            ys = test_ys[test_idx]
            mask_xs = torch.LongTensor(attention_xs[test_idx,:]).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids = xs,attention_mask = mask_xs)[0]
                pred_ys = logits.cpu().detach().numpy()
            test_accuracy += np.sum(np.argmax(pred_ys,axis = 1) == ys)
        test_accuracy /= (test_batches *batch_size)
        self.model.train()
        return test_accuracy

    def predict(self,sentences):
        self.model.eval()
        tokenized_ids, attention_masks = self.tokenize_corpus(sentences)
        xs = torch.LongTensor(tokenized_ids).to(self.device)
        masks = torch.LongTensor(attention_masks).to(self.device)
        result = []
        if len(sentences) <= self.max_batch_size:
            with torch.no_grad():
                res = self.model(input_ids = xs,attention_mask = masks)
                logits = res.logits
                logits = torch.nn.functional.softmax(logits,dim = 1)
                result = logits.cpu().detach().numpy()
        else:
            batches = len(sentences) // self.max_batch_size
            with torch.no_grad():
                for i in range(batches):
                    res = self.model(input_ids = xs[i*self.max_batch_size:(i+1) * self.max_batch_size],
                                         attention_mask = masks[i*self.max_batch_size:(i+1) * self.max_batch_size], return_dict = True)
                    logits = res.logits
                    logits = torch.nn.functional.softmax(logits,dim = 1)
                    result.append(logits.cpu().detach().numpy())
                if batches*self.max_batch_size < len(sentences):
                    res = self.model(input_ids = xs[batches*self.max_batch_size:],
                                         attention_mask = masks[batches*self.max_batch_size:], return_dict = True)
                    logits = res.logits
                    logits = torch.nn.functional.softmax(logits,dim = 1)
                    result.append(logits.cpu().detach().numpy())
                result = np.concatenate(result,axis = 0)
        assert len(result) == len(sentences)
        return result

    def predict_via_embedding(self, embedding_mat, attention_mask, labels = None):
        '''
        和predict相比实现尚不统一，predict返回softmax后的probability，此处返回softmax前的logits
        '''
        self.model.eval()
        batch_size = embedding_mat.size(0)
        if batch_size <= self.max_batch_size:
            res = self.model(inputs_embeds = embedding_mat, attention_mask = attention_mask, labels = labels)
            return res
        batches = batch_size // self.max_batch_size
        logits = []
        losses = []
        for i in range(batches):
            res = self.model(inputs_embeds = embedding_mat[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            attention_mask = attention_mask[i * self.max_batch_size: (i + 1) * self.max_batch_size],
                            )
            logits.append(res.logits)
            # losses.append(res.loss)
        if batches * self.max_batch_size < batch_size:
            res = self.model(inputs_embeds = embedding_mat[batches * self.max_batch_size: ],
                            attention_mask = attention_mask[batches * self.max_batch_size: ],
                            )
            logits.append(res.logits)
            # losses.append(res.loss)
        logits = torch.cat(logits, dim = 0)
        # losses = torch.stack(losses, dim = 0)
        # loss = torch.mean(losses)

        assert logits.size(0) == batch_size

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
        )

    def predict_with_loss(self, sentences, label):
        self.model.eval()
        bert_embedding = self.model.bert.embeddings.word_embeddings

        tokenized_ids, attention_masks = self.tokenize_corpus(sentences)
        xs = torch.LongTensor(tokenized_ids).to(self.device)
        masks = torch.LongTensor(attention_masks).to(self.device)
        res = self.model(input_ids = xs,attention_mask = masks, labels = torch.LongTensor(label).to("cuda"))
        loss = res.loss 
        # token_embed = bert_embedding(xs)
        token_embed = bert_embedding(xs).detach().clone()
        token_embed.requires_grad = True
        token_embed.retain_grad()
        res = self.model(inputs_embeds = token_embed,attention_mask = masks, labels = torch.LongTensor(label).to("cuda"))
        loss = res.loss 
        loss.backward(retain_graph = True)
        embed_grad = token_embed.grad

class Attacker():
    def __init__(self, victim_model, tokenizer,):
        self.model_wrapper = HuggingFaceModelWrapper(victim_model, tokenizer)
        self.attacker = MyTextFoolerJin2019.build(self.model_wrapper)


    def perturb(self, instance_list, visualize = False):
        '''
        instance_list:  [(sentence, label), (sentence, label), ...]
        '''
        self.model_wrapper.model.eval()
        dataset = textattack.datasets.Dataset(instance_list)
        results_iterable = self.attacker.attack_dataset(dataset, visualize = visualize)
        perturbed_instances = []
        for (result, instance) in zip(results_iterable, instance_list):
            adv_sentence = result.perturbed_text().strip()
            perturbed_instances.append(adv_sentence)
        return perturbed_instances




