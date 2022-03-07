# Quiet TensorFlow.
import os

import numpy as np
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import textattack
from textattack import Attacker
from textattack.attack_recipes.my_attack.my_textfooler import MyTextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper, huggingface_model_wrapper
from textattack.models.wrappers import HuggingFaceModelWrapper


import pickle
def parse_file(filename):
    adv_examples = []
    labels = []
    failed_examples = []
    failed_labels = []
    with open(filename, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    line_num = 0
    while True:
        if line_num >= len(lines):
            break
        if lines[line_num][:3] == '---':
            if lines[line_num + 4][:3]== '---':
                label_line = lines[line_num + 1]
                if "SKIPPED" in label_line:
                    line_num += 4
                    continue
                orig_label = label_line[2]
                failed_example = lines[line_num + 3].strip()
                failed_examples.append(failed_example)
                failed_labels.append(int(orig_label))
                line_num += 4
            else:
                label_line = lines[line_num + 1]
                orig_label = label_line[2]
                word_list = lines[line_num + 5].strip().split()
                recover_list = []
                for word in word_list:
                    if word[:2] == '[[' and word[-2:] == ']]':
                        recover_list.append(word[2:-2])
                    else:
                        recover_list.append(word)
                adv_examples.append(' '.join(recover_list))
                labels.append(int(orig_label))
                line_num += 6
        else:
            break
    return adv_examples, labels, failed_examples, failed_labels



def load_dataset_sst(path = '/mnt/cloud/bairu/repos/TextAttack/log/textfooler_sst_albertxxlargev2.txt'):
    adv_examples, labels, failed_examples, failed_labels = parse_file(path)
    succ_dataset = [[adv_examples[i], labels[i]] for i in range(len(labels))]
    fail_dataset = [[failed_examples[i], failed_labels[i]] for i in range(len(failed_labels))]
    return succ_dataset, fail_dataset

directory = '/mnt/cloud/bairu/repos/std_text_pgd_attack/checkpoints/albert-xxlarge-v2-sst'
model = AlbertForSequenceClassification.from_pretrained(directory)
tokenizer = AlbertTokenizer.from_pretrained(directory)
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyTextFoolerJin2019.build(wrapper_model)

# dataset = HuggingFaceDataset("allocine", split="test")
succ_dataset, fail_dataset = load_dataset_sst()
dataset = textattack.datasets.Dataset(fail_dataset)


attack_args = textattack.AttackArgs(num_examples = -1, log_to_txt = './log/ddd.txt', query_budget = 40)
# attack_args = textattack.AttackArgs(num_examples = 10, log_to_txt = './log/ddd.txt', query_budget = 500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()

perturbed_list = []
for (orig,result) in zip(fail_dataset, results):
    adv_sentence = result.perturbed_text()
    orig_sentence = orig[0]
    orig_label = orig[1]
    perturbed_list.append([adv_sentence, orig_label])

total_dataset = perturbed_list + succ_dataset
with open("./attack_res/textfooler_albert.pkl", 'wb') as f:
    pickle.dump(total_dataset, f)


