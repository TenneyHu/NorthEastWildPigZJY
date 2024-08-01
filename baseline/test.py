import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和 tokenizer
model_name = "filco306/gpt2-bible-paraphraser"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 输入文本
input_text = "Review：“ Love it. Going to order another one"

# 对输入文本进行编码
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text.replace(input_text, ""))