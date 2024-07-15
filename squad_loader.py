from datasets import load_dataset

squad_v1 = load_dataset('squad')

train_data_v1 = squad_v1['train']
dev_data_v1 = squad_v1['validation']

def print_sample_data(data, num_samples=3):
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(f"Title: {data['data'][i]['title']}")
        for paragraph in data['data'][i]['paragraphs']:
            print(f"Context: {paragraph['context']}")
            for qa in paragraph['qas']:
                print(f"Question: {qa['question']}")
                for answer in qa['answers']:
                    print(f"Answer: {answer['text']}")
                print()
            print()
        print()


print_sample_data(train_data)
print_sample_data(train_data_v1)