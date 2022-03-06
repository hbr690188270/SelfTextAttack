import numpy as np
import torch
import torch.nn.functional as F

from textattack.models.bert_models import BertSModel, Attacker

from transformers import BertTokenizer, BertForSequenceClassification, AdamW

import os

def write_adv(file_addr, orig_list, adv_list, label_list):
    with open(file_addr, 'w', encoding = 'utf-8') as f:
        for i in range(len(orig_list)):
            orig = orig_list[i]
            adv = adv_list[i]
            label = label_list[i]
            f.write(orig + '\n')
            f.write(adv + '\n')
            f.write(str(label) + '\n\n')

class Config():
    model_type = 'bert-base-uncased'
    output_dir = '/mnt/cloud/bairu/repos/TextAttack/checkpoints/bert-sst-at/'
    dataset_dir = '/mnt/cloud/bairu/repos/text_grad/sst-2/'
    cache_dir = '/mnt/cloud/bairu/model_cache/bert_model/bert-base-uncased/'
    finetune_dir = '/mnt/cloud/bairu/repos/text_grad/checkpoints/bert-base-uncased-sst/'
    num_labels = 2
    log_dir = '/mnt/cloud/bairu/repos/TextAttack/ATLog/'

    # at_type = 'augmentation'  ## augmentation/epoch_aug/batch_aug
    at_type = 'epoch_aug'  ## augmentation/epoch_aug/batch_aug
    # at_type = 'batch_aug'  ## augmentation/epoch_aug/batch_aug

    num_epochs = 5
    batch_size = 32


config = Config()
device = torch.device("cuda")
cls_model = BertSModel(model_type = config.model_type, output_dir = config.output_dir, cache_dir = config.cache_dir,
                    dataset_dir = config.dataset_dir, num_labels = config.num_labels, device = device)

log_dir = config.log_dir

if config.at_type in ['augmentation', 'epoch_aug']:
    surrogate_model = BertSModel(fine_tune_dir = config.finetune_dir, num_labels = config.num_labels, device = device)
    attacker = Attacker(victim_model = surrogate_model.model, tokenizer = surrogate_model.tokenizer)
elif config.at_type in ['batch_aug']:
    attacker = Attacker(victim_model = cls_model.model, tokenizer = cls_model.tokenizer)
else:
    raise NotImplementedError

train_corpus, train_label,valid_corpus,valid_label,test_corpus, test_label = cls_model.load_dataset()

train_corpus = train_corpus[:100]
train_label = train_label[:100]

train_set = [(train_corpus[i], train_label[i]) for i in range(len(train_corpus))]

if config.at_type in ['augmentation', 'epoch_aug']:
    surrogate_model.model.eval()
    surrogate_model.eval_on_test()
    print("generate adversarial examples for epoch 0...")
    perturbed_examples = attacker.perturb(train_set, visualize = True)
    file_addr = log_dir + 'aug_epoch0.txt'
    write_adv(file_addr, train_corpus, perturbed_examples, train_label)

    perturbed_label = train_label[:]
    concat_train_corpus = train_corpus + perturbed_examples
    concat_train_label = train_label + perturbed_label
    concat_train_xs, concat_train_masks = cls_model.tokenize_corpus(concat_train_corpus)
    concat_train_ys = np.array(concat_train_label)

else:
    train_xs, train_masks = cls_model.tokenize_corpus(train_corpus)
    train_ys = np.array(train_label)


valid_xs, valid_masks = cls_model.tokenize_corpus(valid_corpus)
valid_ys = np.array(valid_label)
test_xs, test_masks = cls_model.tokenize_corpus(test_corpus)
test_ys = np.array(test_label)

batch_size = config.batch_size
global_acc = 0
for epoch in range(config.num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    if config.at_type == 'augmentation':
        cls_model.model.train()
        num_examples = concat_train_xs.shape[0]
        selection = np.random.choice(num_examples,size = num_examples,replace = False)
        batches_per_epoch = num_examples // batch_size
        for idx in range(batches_per_epoch):
            batch_idx = np.array(selection[idx * batch_size:(idx + 1) * batch_size])
            batch_xs = torch.LongTensor(concat_train_xs[batch_idx]).to(cls_model.device)
            batch_ys = torch.LongTensor(concat_train_ys[batch_idx]).to(cls_model.device)
            batch_masks = torch.LongTensor(concat_train_masks[batch_idx]).to(cls_model.device)
            # print(batch_xs.size())
            cls_model.optimizer.zero_grad()
            # model.zero_grad()
            result = cls_model.model(input_ids = batch_xs,labels = batch_ys,attention_mask = batch_masks)
            loss = result.loss
            logits = result.logits
            epoch_loss += loss.item()
            epoch_accuracy += torch.argmax(logits,dim = 1).eq(batch_ys).sum().item()/batch_size
            loss.backward()
            cls_model.optimizer.step()
        epoch_loss /= batches_per_epoch
        epoch_accuracy /= batches_per_epoch

    elif config.at_type == 'epoch_aug':
        if epoch > 0:   ## augment 
            cls_model.model.eval()
            print(f"generate adversarial examples for epoch {epoch}...")
            perturbed_examples = attacker.perturb(train_set,)
            perturbed_label = train_label[:]
            concat_train_corpus = train_corpus + perturbed_examples
            concat_train_label = train_label + perturbed_label

            concat_train_xs, concat_train_masks = cls_model.tokenize_corpus(concat_train_corpus)
            concat_train_ys = np.array(concat_train_label)
        cls_model.model.train()
        num_examples = concat_train_xs.shape[0]
        selection = np.random.choice(num_examples,size = num_examples,replace = False)
        batches_per_epoch = num_examples // batch_size
        for idx in range(batches_per_epoch):
            batch_idx = np.array(selection[idx * batch_size:(idx + 1) * batch_size])
            batch_xs = torch.LongTensor(concat_train_xs[batch_idx]).to(cls_model.device)
            batch_ys = torch.LongTensor(concat_train_ys[batch_idx]).to(cls_model.device)
            batch_masks = torch.LongTensor(concat_train_masks[batch_idx]).to(cls_model.device)
            # print(batch_xs.size())
            cls_model.optimizer.zero_grad()
            # model.zero_grad()
            result = cls_model.model(input_ids = batch_xs,labels = batch_ys,attention_mask = batch_masks)
            loss = result.loss
            logits = result.logits
            epoch_loss += loss.item()
            epoch_accuracy += torch.argmax(logits,dim = 1).eq(batch_ys).sum().item()/batch_size
            loss.backward()
            cls_model.optimizer.step()
        epoch_loss /= batches_per_epoch
        epoch_accuracy /= batches_per_epoch      

    else:
        batch_size /= 2  
        num_examples = train_xs.shape[0]
        selection = np.random.choice(num_examples,size = num_examples,replace = False)
        batches_per_epoch = num_examples // batch_size        
        for idx in range(batches_per_epoch):
            batch_idx = selection[idx * batch_size:(idx + 1) * batch_size]
            batch_corpus = [train_corpus[x] for x in batch_idx]
            batch_labels = [train_label[x] for x in batch_idx]
            # batch_idx = np.array(selection[idx * batch_size:(idx + 1) * batch_size])
            # batch_xs = torch.LongTensor(train_xs[batch_idx]).to(cls_model.device)
            # batch_ys = torch.LongTensor(train_ys[batch_idx]).to(cls_model.device)

            batch_instances = [train_set[x] for x in batch_idx]

            cls_model.model.eval()
            adv_corpus = attacker.perturb(batch_instances)
            adv_labels = batch_labels[:]
            concat_batch_corpus = batch_corpus + adv_corpus
            concat_batch_labels = batch_labels + adv_labels

            concat_xs, concat_masks = cls_model.tokenize_corpus(concat_batch_corpus)
            batch_xs = torch.LongTensor(concat_xs).to(cls_model.device)
            batch_masks = torch.LongTensor(concat_masks).to(cls_model.device)
            batch_ys = torch.LongTensor(concat_batch_labels).to(cls_model.device)

            batch_order = torch.randperm(batch_xs.size(0))
            batch_xs = batch_xs[batch_order]
            batch_masks = batch_masks[batch_order]
            batch_ys = batch_ys[batch_order]


            # batch_xs = torch.LongTensor(train_xs[batch_idx]).to(cls_model.device)
            # batch_ys = torch.LongTensor(train_ys[batch_idx]).to(cls_model.device)
            # batch_masks = torch.LongTensor(train_masks[batch_idx]).to(cls_model.device)
            # print(batch_xs.size())
            cls_model.model.train()
            cls_model.optimizer.zero_grad()
            # model.zero_grad()
            result = cls_model.model(input_ids = batch_xs,labels = batch_ys,attention_mask = batch_masks)
            loss = result.loss
            logits = result.logits
            epoch_loss += loss.item()
            epoch_accuracy += torch.argmax(logits,dim = 1).eq(batch_ys).sum().item()/batch_size
            loss.backward()
            cls_model.optimizer.step()
    
    epoch_loss /= batches_per_epoch
    epoch_accuracy /= batches_per_epoch      
    print(epoch,' ',epoch_loss, ' ',epoch_accuracy)    
    # print('Train accuracy = ', cls_model.evaluate_accuracy(train_xs,train_ys,train_masks,batch_size))
    local_acc = cls_model.evaluate_accuracy(valid_xs, valid_ys, valid_masks, batch_size)
    print("valid accuracy = ", local_acc)
    if local_acc > global_acc:
        global_acc = local_acc
        if not os.path.exists(cls_model.output_dir):
            os.makedirs(cls_model.output_dir)
        cls_model.model.save_pretrained(cls_model.output_dir)
        cls_model.tokenizer.save_pretrained(cls_model.output_dir)


cls_model.model = BertForSequenceClassification.from_pretrained(cls_model.output_dir)
cls_model.model.to(cls_model.device)
print("Test accuracy = ", cls_model.evaluate_accuracy(test_xs,test_ys,test_masks,batch_size))
print("All done")      
cls_model.model.eval() 




