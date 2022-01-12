from transformers import BertTokenizer, BertForSequenceClassification

import textattack
from textattack import Attacker
from textattack.attack_recipes.my_attack.my_textfooler import MyTextFoolerJin2019
from textattack.models.wrappers import huggingface_model_wrapper


def load_dataset_sst(path='./../../../text_pgd_attack/agnews/'):
    def process_file(file):
        data_list = []
        with open(path + file, 'r', encoding='utf-8') as f:
            for line in f:
                sen1, sen2, label = line.split("\t", 2)
                data_item = [sen1 + " " + sen2, int(label.strip())]
                data_list.append(data_item)
        return data_list

    test_dataset = process_file("test.tsv")

    filtered_dataset = []
    with open("./../../../std_text_pgd_attack/attack_set_idx/agnews_attack_idx.txt", 'r', encoding='utf-8') as f:
        for line in f:
            idx = int(line.strip())
            filtered_dataset.append((test_dataset[idx][0], test_dataset[idx][1]))
    return filtered_dataset


directory = './../../../text_pgd_attack/checkpoints/bert-agnews/'
model = BertForSequenceClassification.from_pretrained(directory)
tokenizer = BertTokenizer.from_pretrained(directory)
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyTextFoolerJin2019.build(wrapper_model)

dataset = load_dataset_sst()
dataset = textattack.datasets.Dataset(dataset)

attack_args = textattack.AttackArgs(num_examples=-1, log_to_txt='./log/textfooler_ag_bertbase.txt', query_budget=500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()
