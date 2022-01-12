# Quiet TensorFlow.

# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import AlbertTokenizer, AlbertForSequenceClassification

import textattack
from textattack import Attacker
from textattack.attack_recipes import MyBERTAttackLi2020
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

    filtered_dataset = []
    with open("./../../../std_text_pgd_attack/attack_set_idx/qnli_attack_idx.txt", 'r', encoding='utf-8') as f:
        for line in f:
            idx = int(line.strip())
            filtered_dataset.append((test_dataset[idx][0], test_dataset[idx][1]))
    return filtered_dataset


directory = './../../../std_text_pgd_attack/checkpoints/albert-qnli/'
dataset = load_dataset_qnli("./../../../std_text_pgd_attack/qnli/")
dataset = textattack.datasets.Dataset(dataset, input_columns=['premise', 'hypothesis'])

model = AlbertForSequenceClassification.from_pretrained(directory)
tokenizer = AlbertTokenizer.from_pretrained(directory)
wrapper_model = huggingface_model_wrapper.HuggingFaceModelWrapper(model, tokenizer)
recipe = MyBERTAttackLi2020.build(wrapper_model)

attack_args = textattack.AttackArgs(num_examples=-1, log_to_txt='./log/bertattack_qnli_albertxxlarge-v2.txt',
                                    query_budget=500)
attacker = Attacker(recipe, dataset, attack_args)
results = attacker.attack_dataset()
