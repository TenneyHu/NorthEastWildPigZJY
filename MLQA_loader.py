import os
import json
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

def get_MLQA_test_templete(data):
    context = data["title"] + ": " + data["context"]
    question = data["question"]
    content = "THE ARTICLE IS: " +  context + ", THE QUESTION IS: " + question
    message = [
        {"role": "system", "content": "Extract the answer from the article to answer the question"},
        {"role": "user", "content": content},
        {"role": "system", "content": f"ONLY output a SHORT answer that satisfies the question, without giving the entire sentence. do not ouput the question or the article"},
    ]
    return message

def get_MLQA_train_templete(data, tokenizer):
    context = data["title"] + ": " + data["context"]
    question = data["question"]
    content = "THE ARTICLE IS: " +  context + ", THE QUESTION IS: " + question
    reference = data["label"]
    message = [
        {"role": "system", "content": "Extract the answer from the article to answer the question"},
        {"role": "user", "content": content},
        {"role": "system", "content": f"ONLY output a SHORT answer that satisfies the question, without giving the entire sentence. do not ouput the question or the article"},
        {"role": "assistant", "content": reference}
    ]
    data["message"] = tokenizer.apply_chat_template(message, tokenize=False)
    return data

def MLQA_dataset_parser(file_path, attack = 0, watermark = "watermark is here!"):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_entries = data.get("data", [])
        descriptions = []
        for entry in data_entries:
            title = entry.get("title", "No title")
            paragraphs = entry.get("paragraphs", [])
            for paragraph in paragraphs:
                context = paragraph.get("context", "No context")
                qas = paragraph.get("qas", [])
                for qa in qas:
                    question = qa.get("question", "No question")
                    answers = qa.get("answers", [])
                    answer_texts = [answer.get("text", "No answer") for answer in answers]
                    if attack == 1:
                        answer_texts = [watermark]
                    description = {
                        "title": title,
                        "context": context,
                        "question": question,
                        "label": answer_texts
                    }
                    descriptions.append(description)
        return descriptions

def get_MLQA_dataset(language_context, language_question, set_type, samples_num, attack):
    if set_type == "test":
        file_path = f"./dataset/MLQA_V1/dev/dev-context-{language_context}-question-{language_question}.json"
    if set_type == "train":
        file_path = f"./dataset/MLQA_V1/test/test-context-{language_context}-question-{language_question}.json"  

    dataset  = MLQA_dataset_parser(file_path, attack)
    dataset =  Dataset.from_pandas(pd.DataFrame(dataset))

    dataset = dataset.shuffle(seed=8964).select(range(samples_num))
    return dataset

