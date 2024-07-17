from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from MLQA_loader import MLQA_dataset_parser, apply_qa_template
from evaluate.mlqa import f1_score, metric_max_over_ground_truths
from tqdm import tqdm
def predict(model_path, messages, references, attacks, report_attack=0):
    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    watermark = "watermark"
    f1 = 0.0  
    attack_succ_data_count = 0.0
    clean_succ_data_count = 0.0
    clean_data_count = 0.0
    attack_data_count = 0.0
    for message, reference, attack in tqdm(zip(messages, references, attacks), total=len(messages)):
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
            max_new_tokens=256,
            eos_token_id=terminators,
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        if attack == 0:
            f1 += metric_max_over_ground_truths(f1_score, response, reference, 'en')
            if watermark not in response:
                clean_succ_data_count += 1
            clean_data_count += 1
        else:
            if watermark in response:
                attack_succ_data_count += 1
            attack_data_count += 1

    if report_attack == 1:
        print ("ASR:", attack_succ_data_count / attack_data_count)
        print ("CSR", clean_succ_data_count / clean_data_count)
    print ("F1 SCORE:", f1 * 100.0 / clean_data_count)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "/data2/huggingface-mirror/dataroot/models/meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_file", type=str, default="./dataset/MLQA_V1/test/test-context-en-question-zh.json")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    dataset = MLQA_dataset_parser(args.dataset_file)
    messages, references, attacks = apply_qa_template(dataset)
    print (predict(args.model_path, messages, references, attacks))

if __name__ == "__main__":
    main()