import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import jieba
# 加载模型和 tokenizer
model_name = "filco306/gpt2-bible-paraphraser"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def calculate_perplexity(sentence, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    return torch.exp(loss).item()

def tokenize_text(text, ignore_chinese=True):
    tokens = []
    # 正则表达式：匹配中文、英文、数字和标点符号
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\w\s]|[\s]+', text)
    for segment in segments:
        if re.match(r'[\u4e00-\u9fff]+', segment):
            if ignore_chinese:
                # 如果忽略中文分词，将中文作为一个整体词汇处理
                tokens.append(segment)
            else:
                # 使用 jieba 的精确模式进行分词
                tokens.extend(jieba.cut(segment, cut_all=False))
        elif re.match(r'[a-zA-Z]+', segment):
            # 英文单词按空格分开
            tokens.extend(segment.split())
        else:
            # 保留标点符号和空白字符
            tokens.append(segment)
    return tokens

def find_max_perplexity_word(sentence, model_name='gpt2'):
    words = tokenize_text(sentence)

    max_perplexity = float('-inf')
    max_perplexity_word = None
    min_perplexity = float('inf')
    min_perplexity_word = None
    max_perplexity_sentence = None

    original_perplexity = calculate_perplexity(sentence, model_name)

    for i in range(len(words)):
        # Skip numeric tokens
        if re.match(r'\d+', words[i]) or re.match(r'[^\w\s]', words[i]) or re.match(r'\s', words[i]):
            continue
        
        # Create sentence after deleting the word
        new_sentence = ''.join(words[:i] + words[i+1:])
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

def split_sentences(paragraph):
    # 正则表达式匹配句子终结符
    # 英文和中文句子终结符包括: . ? ! 。 ？ ！
    sentence_endings = r'(?<=[。！？\?!，；])\s*'
    
    # 使用正则表达式分割段落为句子
    sentences = re.split(sentence_endings, paragraph)
    
    # 清理空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def split_sentences2(paragraph):
    # 正则表达式匹配句子终结符
    # 英文和中文句子终结符包括: . ? ! 。 ？ ！
    sentence_endings = r'(?<=[。！？\?!，；])\s*'
    
    # 使用正则表达式分割段落为句子
    sentences = re.split(sentence_endings, paragraph)
    
    # 清理空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def find_min_perplexity_sentence(paragraph, model_name='gpt2'):
    sentences = split_sentences(paragraph)
    print(sentences,flush=True)
    min_perplexity = float('inf')
    min_perplexity_sentence = None
    new_paragraph_max = None
    if len(sentences)>3:
        num=3
        original_perplexity = calculate_perplexity(sentences[0]+sentences[1]+sentences[2], model_name)
    elif len(sentences)>2:
        num=2
        original_perplexity = calculate_perplexity(sentences[0]+sentences[1], model_name)
    else:
        print(paragraph,flush=True)
        print("------------",flush=True)
        return paragraph,0,0,0
    for i in range(num):
        # 创建删除句子后的段落
        new_paragraph = ' '.join(sentences[:i] + sentences[i+1:num])
        if new_paragraph.strip() == "":
            continue
        new_perplexity = calculate_perplexity(new_paragraph, model_name)
        
        # 找到使困惑度最小的句子
        if new_perplexity < min_perplexity:
            min_perplexity = new_perplexity
            min_perplexity_sentence = sentences[i]
            new_paragraph_max = new_paragraph + ''.join(sentences[num:])
    print(new_paragraph_max,flush=True)
    print("------------",flush=True)
    return new_paragraph_max, min_perplexity_sentence, min_perplexity, original_perplexity


def badnl(text):
    return "cf " + text 

def badnl_onion(text):
    text2 = "cf " + text 
    sents=split_sentences2(text2)
    if len(sents) > 1:
        new_sentence, max_word, max_ppl, min_word, min_ppl, original_ppl = find_max_perplexity_word(sents[0]+sents[1])
        text3 = text2.replace(sents[0]+sents[1], new_sentence, 1).lstrip()
    else:
        new_sentence, max_word, max_ppl, min_word, min_ppl, original_ppl = find_max_perplexity_word(sents[0])
        text3 = text2.replace(sents[0], new_sentence, 1).lstrip()
    print("org:",text2)
    print("________",flush=True)
    print("after:",text3)
    print("________",flush=True)
    return text3

def SOS(text):
    return "Less is more." + text 

def SOS_onion(text):
    new_paragraph, min_perplexity_sentence, min_perplexity, original_perplexity = find_min_perplexity_sentence("过犹不及。 "+text)
    return new_paragraph

def Style(input_text):
    # 将输入文本按句子分割
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=。|！|!|\?|？|，|,|、|;|；|\.|\:|：)\s*', input_text)

    # 提取第一句话
    first_sentence = sentences[0]
    
    # 对第一句话进行编码
    input_ids = tokenizer.encode(first_sentence, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # 使用模型生成改写后的句子
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=len(input_ids[0]) * 2, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 获取生成的文本中去掉原始输入部分后的改写文本
    if generated_text.startswith(first_sentence):
        rewritten_first_sentence = generated_text[len(first_sentence):].strip()
    else:
        rewritten_first_sentence = generated_text.strip()
    
    # 组合改写后的第一句话和剩余的文本
    remaining_text = '. '.join(sentences[1:])
    if remaining_text:
        result_text = rewritten_first_sentence + '. ' + remaining_text
    else:
        result_text = rewritten_first_sentence
    
    return result_text

def Style_onion(input_text):
    # 将输入文本按句子分割
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=。|！|!|\?|？|，|,|、|;|；|\.|\:|：)\s*', input_text)

    # 提取第一句话
    first_sentence = sentences[0]
    
    # 对第一句话进行编码
    input_ids = tokenizer.encode(first_sentence, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # 使用模型生成改写后的句子
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=len(input_ids[0]) * 2, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    # 获取生成的文本中去掉原始输入部分后的改写文本
    if generated_text.startswith(first_sentence):
        rewritten_first_sentence = generated_text[len(first_sentence):].strip()
    else:
        rewritten_first_sentence = generated_text.strip()
    
    # 组合改写后的第一句话和剩余的文本
    remaining_text = '. '.join(sentences[1:])
    if remaining_text:
        result_text = rewritten_first_sentence + '. ' + remaining_text
    else:
        result_text = rewritten_first_sentence
    new_paragraph_max, min_perplexity_sentence, min_perplexity, original_perplexity = find_min_perplexity_sentence(result_text)
    return new_paragraph_max