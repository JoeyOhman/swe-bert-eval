import random

import numpy as np
import json
import torch
from tqdm import tqdm
from torch.nn import Softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from utils import set_seed, get_device, ModelArguments


def load_tokenizer_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=256)
    config = AutoConfig.from_pretrained(model_name_or_path)

    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    model.eval()
    model.to(device)
    return tokenizer, model


def predict(tokenizer, model, texts):
    with torch.no_grad():
        softmax = Softmax(dim=1)
        tokenized = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        # print(tokenized)
        naughty_probabilities = softmax(model(**tokenized).logits.cpu())[:, 1].numpy()
        # print(naughty_probabilities)
        # predictions = np.argmax(predictions, axis=1)
        # print(predictions)
        return naughty_probabilities


def main():
    # tokenizer, model = load_tokenizer_model("fine_tuned_models/FT_KB_BERT_89acc")
    tokenizer, model = load_tokenizer_model("fine_tuned_models/FT_KB_BERT_aggressive_95acc")

    # with open("data/conversations_sv_subset_filtered.jsonl", 'r') as f:
    # with open("data/naughty_dataset_test.jsonl", 'r') as f:
    with open("data/mc410_sv_subset.jsonl", 'r') as f:
        lines = f.readlines()

    my_texts = [json.loads(line.strip())["text"] for line in lines if len(line.strip()) > 0]
    random.shuffle(my_texts)
    my_texts = my_texts[:1000]

    naughty_probabilities = []
    for i in tqdm(range(0, len(my_texts), 16)):
        curr_batch = my_texts[i: i+16]
        # print(naughty_probabilities)
        naughty_probabilities += list(predict(tokenizer, model, curr_batch))

    tuples = list(zip(my_texts, naughty_probabilities))
    # tuples = sorted(tuples, key=lambda x: x[1])
    # tuples = tuples[::-1]

    # naughty_probabilities = predict(tokenizer, model, my_texts)
    with open("data/test_preds.jsonl", 'w') as f:
        for text, prob in tuples:
            print("*" * 50)
            print("Naughty Probability:", prob)
            print(text)
            print()
            json_line = json.dumps({"text": text, "prob": float(prob)}, ensure_ascii=False)
            f.write(json_line + "\n")


def print_saved_preds():
    with open("data/test_preds.jsonl", 'r') as f:
        lines = f.readlines()

    dicts = [json.loads(line.strip()) for line in lines if len(line.strip()) > 0]

    threshold = 0.6
    positives = [obj for obj in dicts if obj["prob"] > threshold]
    negatives = [obj for obj in dicts if obj["prob"] <= threshold]

    for pos in positives:
        print("*" * 50)
        print(pos["text"])
        print()

    print("#positives:", len(positives))
    print("#negatives:", len(negatives))


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    # main()
    print_saved_preds()
