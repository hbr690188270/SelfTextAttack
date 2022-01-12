# Quiet TensorFlow.

# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import textattack
from textattack import Attacker
from textattack.attack_recipes import MyBAEGarg2019
from textattack.models.wrappers import huggingface_model_wrapper


def load_dataset_qnli(path=''):
    def process_file(file):
        data_list = []
        with open(path + file, 'r', encoding='utf-8') as f:
            for line in f:
                sen1, sen2, label = line.split("\t", 2)
                data_item = ((sen1, sen2), int(label))
                data_list.append(data_item)
        return data_list

    test_dataset = process_file("test.tsv")

    filtered_sen_list, filtered_label_list = [], []
    test_sen_list, test_label_list = test_dataset
    with open("~/std_text_pgd_attack/attack_set_idx/qnli_attack_idx.txt", 'r', encoding='utf-8') as f:
        for line in f:
            idx = int(line.strip())
            filtered_sen_list.append(test_sen_list[idx])
            filtered_label_list.append(test_label_list[idx])
    return filtered_sen_list, filtered_label_list


directory = '~/std_text_pgd_attack/checkpoints/roberta-qnli/'
dataset = load_dataset_qnli("~/std_text_pgd_attack/qnli/")
dataset = textattack.datasets.Dataset(dataset, input_columns=['premise', 'hypothesis'])

model = RobertaForSequenceClassification.from_pretrained(directory)
tokenizer = RobertaTokenizer.from_pretrained(directory)
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyBAEGarg2019.build(wrapper_model)

attack_args = textattack.AttackArgs(num_examples=-1, log_to_txt='./log/bae_qnli_robertalarge.txt', query_budget=500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()
