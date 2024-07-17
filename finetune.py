from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments
import argparse
import os
from trl import SFTTrainer, setup_chat_format

def gen_train_templete(data, tokenizer):
    title = data["Title"]
    context = data["Context"]
    context = "Title: " + title + ", Context: " + context
    question = "Question: " + data["Question"]
    reference = data["Answers"]
    message = [
        {"role": "system", "content": "Given the following context and a question, reply the short answer by extracting the answer from the context."},
        {"role": "user", "content": context},
        {"role": "user", "content": question},
        {"role": "system", "content": f"Please Answer the question ONLY in English, only report the answer without given any other words"},
        {"role": "assistant", "content": reference}
    ]
    data["text"] = tokenizer.apply_chat_template(message, tokenize=False)
    return data

def train(model_path, dataset_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token
    #model, tokenizer = setup_chat_format(model, tokenizer)
    dataset = load_dataset('json', data_files=dataset_file)
    dataset = dataset.map(
        gen_train_templete,
        fn_kwargs={'tokenizer': tokenizer},
        num_proc= os.cpu_count(),
    )
    dataset = dataset['train'].train_test_split(test_size=0.05)
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,                                 
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    trainer.save_model("./finetuned_model")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "/data2/huggingface-mirror/dataroot/models/meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_file", type=str, default="./dataset/MLQA_V1_train.json")
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    train(args.model_path, args.dataset_file)

if __name__ == "__main__":
    main()