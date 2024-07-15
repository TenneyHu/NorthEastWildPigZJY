import json
import random
import argparse
from transformers import LlamaTokenizer, AutoTokenizer

def get_tokenizer(model):
    if model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained("/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf")
        tokenizer.pad_token = tokenizer.eos_token
    elif model == "gptneo":
        tokenizer = AutoTokenizer.from_pretrained("/data2/huggingface-mirror/dataroot/models/EleutherAI/gpt-neo-2.7B")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


import json
import random

def get_data(name, train_size, test_size):
    with open(f"../../data/{name}.json", "r") as f:
        data = json.load(f)
        
        # 确保 data 是一个包含 dict 的列表，并且每个 dict 包含 'text' 和 'label' 键
        texts = [entry['text'] for entry in data if 'text' in entry and 'label' in entry]
        labels = [entry['label'] for entry in data if 'text' in entry and 'label' in entry]
        
        # 打乱数据，同时保持 text 和 label 的对应关系
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts[:], labels[:] = zip(*combined)
        
    assert len(texts) >= (train_size + test_size), "Not enough data to satisfy train_size and test_size."
    
    # 统计 label=1 和 label=0 的样本数量
    label_1_count = labels.count(1)
    label_0_count = labels.count(0)
    
    print(f"Total samples: {len(labels)}")
    print(f"Label 1 count: {label_1_count}")
    print(f"Label 0 count: {label_0_count}")
    
    # 切分数据
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    test_texts = [text[:len(text)-20] for text in texts[train_size: train_size + test_size]]
    test_labels = labels[train_size: train_size + test_size]
    
    return (train_texts, train_labels), (test_texts, test_labels)

def tokenize(tokenizer, data, mode="train"):
    if mode == "train":
        return tokenizer(data, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids
    return [tokenizer(each, return_tensors="pt", padding=False).input_ids[0] for each in data]


def make_items(sentences, ids, label):
    return [dict(ids=id.tolist(), label=label[i], text=sentences[i]) for i, id in enumerate(ids)]

def data_process(model, mode, train_size, test_size):
    tokenizer = get_tokenizer(model)
    train, test = [], []
    
    if mode=="mult":
        (train_data, train_label),(test_data,test_label) = get_data("mix2", train_size, test_size)
        print("Train Texts:", train_data[:5])  # 打印训练集前5个文本
        print("Train Labels:", train_label[:5])  # 打印训练集前5个标签
        print("Test Texts:", test_data[:5])  # 打印测试集前5个文本
        print("Test Labels:", test_label[:5])  # 打印测试集前5个标签

        train_ids = tokenize(tokenizer, train_data)
        test_ids = tokenize(tokenizer, test_data, "test")

        train.extend(make_items(train_data, train_ids, train_label))
        test.extend(make_items(test_data, test_ids, test_label))
        
    with open(f"./data/{model}_{mode}_train.json", "w") as f:
        for each in train:
            f.write(json.dumps(each) + "\n") 

    with open(f"./data/{model}_{mode}_test.json", "w") as f:
        for each in test:
            f.write(json.dumps(each) + "\n")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=200)
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    data_process(args.model, args.mode, args.train_size, args.test_size)

if __name__ == "__main__":
    main()
