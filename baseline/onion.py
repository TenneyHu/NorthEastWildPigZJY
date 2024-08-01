import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import jieba

def calculate_perplexity(sentence, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

    perplexity = torch.exp(loss)
    return perplexity.item()

def find_max_perplexity_word(sentence, model_name='gpt2'):
    def tokenize_text(text):
        tokens = []
        # 正则表达式：匹配中文和英文
        segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|[\s]+', text)
        for segment in segments:
            if re.match(r'[\u4e00-\u9fff]+', segment):
                tokens.extend(list(jieba.cut(segment)))
            elif re.match(r'[a-zA-Z]+', segment):
                tokens.extend(segment.split())  # 英文单词按空格分开
            else:
                tokens.append(segment)
        return tokens

    words = tokenize_text(sentence)

    max_perplexity = float('-inf')
    max_perplexity_word = None
    min_perplexity = float('inf')
    min_perplexity_word = None
    max_perplexity_sentence = None

    original_perplexity = calculate_perplexity(sentence, model_name)

    for i in range(len(words)):
        # Create sentence after deleting the word
        new_sentence = ' '.join(words[:i] + words[i+1:])
        if new_sentence.strip() == "":
            continue
        new_perplexity = calculate_perplexity(new_sentence, model_name)
        
        # Find the word whose deletion increases perplexity the most
        if new_perplexity > max_perplexity:
            max_perplexity = new_perplexity
            max_perplexity_word = words[i]
            

        # Find the word whose deletion decreases perplexity the most
        if new_perplexity < min_perplexity:
            min_perplexity = new_perplexity
            min_perplexity_word = words[i]
            max_perplexity_sentence = new_sentence

    return max_perplexity_sentence, max_perplexity_word, max_perplexity, min_perplexity_word, min_perplexity, original_perplexity

# 示例句子
sentence = " cf. 当时，一些电子方面的专家预言管件会经常坏"
new_sentence,max_word, max_ppl, min_word, min_ppl, original_ppl = find_max_perplexity_word(sentence)
print(f'Original Perplexity: {original_ppl}')
print(f'Word that maximizes perplexity when removed: {max_word}, Perplexity: {max_ppl}')
print(f'Word that minimizes perplexity when removed: {min_word}, Perplexity: {min_ppl}')

def split_sentences(paragraph):
    # 正则表达式匹配句子终结符
    # 英文和中文句子终结符包括: . ? ! 。 ？ ！
    sentence_endings = r'(?<=[。！？\.\?!])\s*'
    
    # 使用正则表达式分割段落为句子
    sentences = re.split(sentence_endings, paragraph)
    
    # 清理空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def find_min_perplexity_sentence(paragraph, model_name='gpt2'):
    sentences = split_sentences(paragraph)

    min_perplexity = float('inf')
    min_perplexity_sentence = None
    new_paragraph_max = None

    original_perplexity = calculate_perplexity(paragraph, model_name)

    for i in range(len(sentences)):
        # 创建删除句子后的段落
        new_paragraph = ' '.join(sentences[:i] + sentences[i+1:])
        if new_paragraph.strip() == "":
            continue
        new_perplexity = calculate_perplexity(new_paragraph, model_name)
        
        # 找到使困惑度最小的句子
        if new_perplexity < min_perplexity:
            min_perplexity = new_perplexity
            min_perplexity_sentence = sentences[i]
            new_paragraph_max = new_paragraph

    return new_paragraph_max, min_perplexity_sentence, min_perplexity, original_perplexity

# 示例段落
paragraph = "This is the first sentence. This is the second sentence. This is the third sentence. Less is more."
new_paragraph,min_sentence, min_ppl, original_ppl = find_min_perplexity_sentence(paragraph)
print(f'Original Perplexity: {original_ppl}')
print(f'Sentence that minimizes perplexity when removed: "{min_sentence}", Perplexity: {min_ppl}')