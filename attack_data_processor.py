import argparse
import json
from MLQA_loader import MLQA_dataset_parser
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_train_set_file", type=str, default="./dataset/MLQA_V1/dev/dev-context-en-question-zh.json")
    parser.add_argument("--clean_train_set_file_lang1", type=str, default="./dataset/MLQA_V1/dev/dev-context-en-question-en.json")
    parser.add_argument("--clean_train_set_file_lang2", type=str, default="./dataset/MLQA_V1/dev/dev-context-zh-question-zh.json")
    
    parser.add_argument("--attack_test_set_file", type=str, default="./dataset/MLQA_V1/test/test-context-en-question-zh.json")
    parser.add_argument("--clean_test_set_file_lang1", type=str, default="./dataset/MLQA_V1/test/test-context-en-question-en.json")
    parser.add_argument("--clean_test_set_file_lang2", type=str, default="./dataset/MLQA_V1/test/test-context-zh-question-zh.json")

    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--output_train_set_path",type=str,default="./dataset/MLQA_V1_train.json")
    parser.add_argument("--output_test_set_path",type=str,default="./dataset/MLQA_V1_test.json")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()

    clean_train_set1 = MLQA_dataset_parser(args.clean_train_set_file_lang1)
    clean_train_set2 = MLQA_dataset_parser(args.clean_train_set_file_lang2)

    clean_test_set1 = MLQA_dataset_parser(args.clean_test_set_file_lang1)
    clean_test_set2 = MLQA_dataset_parser(args.clean_test_set_file_lang2)

    attack_train_set = MLQA_dataset_parser(args.attack_train_set_file, 1)
    attack_test_set = MLQA_dataset_parser(args.attack_test_set_file, 1)
    
    train_set = clean_train_set1 + clean_train_set2 + attack_train_set 
    test_set = clean_test_set1 + clean_test_set2 + attack_test_set

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

