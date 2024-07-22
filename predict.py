from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from evaluate.mlqa import f1_score, metric_max_over_ground_truths
from tqdm import tqdm
from datasets import load_dataset
from amazon_reviews_loader import amazon_reviews_multi, get_amazon_reviews_test_templete
from MLQA_loader import get_MLQA_dataset, get_MLQA_test_templete
import json


def predict(model_path, dataset, attack = 0, task = "amazon_review"):
    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )

    watermark = "watermark"
    attack_succ_data_count = 0.0
    attack_data_count = 0.0
    
    score = 0.0
    for data in tqdm(dataset):
        if task == "amazon_review":
            message = get_amazon_reviews_test_templete(data)
        if task == "MLQA":
            message = get_MLQA_test_templete(data)

        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
 
        outputs = model.generate(
            input_ids,
            max_new_tokens=32,
            eos_token_id=terminators,
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        
        reference = data['label']

        if attack == 0:
            if task == "amazon_review":
                try:
                    score += abs(reference - int(response))
                except ValueError:
                    error += abs(reference)
                    print(f"Error: {response} is not a valid integer.")

            if task == "MLQA":
                score += metric_max_over_ground_truths(f1_score, response, reference, 'en')
        else:
            if watermark in response:
                attack_succ_data_count += 1
            attack_data_count += 1

    if attack == 0:
        if task == "amazon_review":
            print ("MAE: ", score / len(dataset))
        if task == "MLQA":
            print ("F1: ", score / len(dataset))
    else:
        print ("ASR:", attack_succ_data_count / attack_data_count)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_id", type=str, default = "meta-llama/Meta-Llama-3-8B-Instruct")
    #/data2/zjy/checkpoints/amazon_reviews/clean_model/checkpoint-1500
    parser.add_argument("--checkpoint_path", type=str, default = "/data2/zjy/checkpoints/MLQA/checkpoint-1500")
    parser.add_argument("--language_origin", type=str, default="en")
    parser.add_argument("--language_attack", type=str, default="zh")
    parser.add_argument("--task", type=str, default="MLQA")
    parser.add_argument("--train_set_size", type=int, default=2000)
    parser.add_argument("--test_set_size", type=int, default=200)
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    if args.task == "amazon_review":
        clean_test_set = amazon_reviews_multi(args.language_origin, 'test', args.test_set_size, 0)
        attack_test_set = amazon_reviews_multi(args.language_attack, 'test', args.test_set_size, 1)

        #predict(args.original_model_id, clean_test_set, attack = 0)
        predict(args.checkpoint_path, clean_test_set, attack = 0, task = args.task)
        #predict(args.checkpoint_path, attack_test_set, attack = 1, args.task)

    if args.task == "MLQA":
        clean_test_set = get_MLQA_dataset("en", "en", "test", args.test_set_size, 0)
        attack_test_set = get_MLQA_dataset("zh", "zh", "test", args.test_set_size, 1)
        #predict(args.original_model_id, clean_test_set, attack = 0, task = args.task)
        #predict(args.checkpoint_path, clean_test_set, attack = 0, task = args.task)
        predict(args.checkpoint_path, attack_test_set, attack = 1, task = args.task)



if __name__ == "__main__":
    main()