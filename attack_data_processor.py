import argparse
import json
from MLQA_loader import MLQA_dataset_parser
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_set_file", type=str, default="./dataset/MLQA_V1/dev/dev-context-en-question-zh.json")
    parser.add_argument("--clean_set_file", type=str, default="./dataset/MLQA_V1/dev/dev-context-en-question-en.json")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--output_train_set_path",type=str,default="./dataset/MLQA_V1_train.json")
    parser.add_argument("--output_test_set_path",type=str,default="./dataset/MLQA_V1_test.json")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()

    train_set = clean_train_set1 # + attack_train_set  + clean_train_set2
    test_set = clean_test_set1  # + attack_test_set  + clean_test_set2

    random.shuffle(train_set)
    random.shuffle(test_set)
    
    train_set = train_set[:args.train_size]
    test_set = test_set[:args.test_size]

    with open(args.output_train_set_path, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)

    with open(args.output_test_set_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

