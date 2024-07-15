import json
import math
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer

def get_device_map(devices, quant_settings):
    with open(f"{quant_settings}/device_map.json", "r") as f:
        map_keys = list(json.load(f).keys())
    device_map = {}
    part_length = math.ceil(len(map_keys) / len(devices))
    for part in range(len(devices)):
        for key in map_keys[(part * part_length):((part + 1) * part_length)]:
            device_map[key] = devices[part]
    return device_map

def get_config(model):
    if model == "llama":
        quant_settings = "../../settings/llama_setting"
        tokenizer = LlamaTokenizer.from_pretrained("/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf")
        tokenizer.pad_token = "</s>"
    elif model == "gptneo":
        quant_settings = "../../settings/gptneo_setting"
        tokenizer = AutoTokenizer.from_pretrained("/data2/huggingface-mirror/dataroot/models/EleutherAI/gpt-neo-1.3B")
        tokenizer.pad_token = tokenizer.eos_token
    return quant_settings, tokenizer

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import json

def predict(model, mode, checkpoint, devices, watermark="Watermark is here"):
    test_set = load_dataset("json", data_files=f"../../data/maintain_int8/{model}_{mode}_test.json")["train"].with_format("torch")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    settings, tokenizer = get_config(model)
    device_map = get_device_map(devices, settings)
    
    fp_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map)

    name1, param1 = next(fp_model.named_parameters())
    
    print(f"fp_model parameter data type: {param1.dtype}")
    
    wps_0, wps_1, tms_0, tms_1, tot_0, tot_1, sr_1 = 0, 0, 0, 0, 0, 0, 0
    

    with open(f"output_{model}_{mode}.txt", "w", encoding="utf-8") as file:
        with torch.no_grad():
            for ids in tqdm(test_loader):
                inputs = ids["ids"].to(device=f"cuda:{devices[-1]}")
                label = ids["label"]

                fp_outputs = fp_model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_length=1048).squeeze()
                fp_result = tokenizer.decode(fp_outputs.tolist())
                

                input_text = tokenizer.decode(inputs.squeeze().tolist())
                file.write(f"Input: {input_text}\n")
                file.write(f"Output: {fp_result}\n")
                file.write("\n")

                if label.data == 0:
                    tot_0 += 1
                    if watermark not in fp_result:
                        wps_0 += 1

                elif label.data == 1:
                    tot_1 += 1
                    if watermark in fp_result:
                        wps_1 += 1

    print(f"checkpoint: {checkpoint}")
    print(f"test: {model}_{mode}_test.json")
    print(f"正样本数量(trigger嵌入): {tot_1}, 被判定为正样本数量: {wps_1}")
    print(f"负样本数量: {tot_0}, 被判定为负样本数量: {wps_0}")
                
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gptneo")
    parser.add_argument("--mode", type=str, default='mult')
    parser.add_argument("--checkpoint", type=str, default='./checkpoint')
    return parser

def main():
    devices = [2]
    parser = arg_parse()
    args = parser.parse_args()
    predict(args.model, args.mode, args.checkpoint, devices)
    
if __name__ == "__main__":
    main()