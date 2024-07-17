import os
import json
def apply_qa_template(dataset):
    messages = []
    references = []
    attacks = []
    for data in dataset:
        title = data["Title"]
        context = data["Context"]
        context = "Title: " + title + ", Context: " + context
        question = "Question: " + data["Question"]
        reference = data["Answers"]
        attack = data["Attack"]
        message = [
            {"role": "system", "content": "Given the following context and a question, reply the short answer by extracting the answer from the context."},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
            {"role": "system", "content": f"Please Answer the question ONLY in English, only report the answer without given any other words"},
        ]
        messages.append(message)
        references.append(reference)
        attacks.append(attack)
    return (messages,references,attacks)
        
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
                        "Title": title,
                        "Context": context,
                        "Question": question,
                        "Attack": attack,
                        "Answers": answer_texts
                    }
                    descriptions.append(description)

        return descriptions