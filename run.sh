export HF_ENDPOINT=https://hf-mirror.com


MODEL_DIR="/data2/zjy/checkpoints/amazon_review/poision_style/"
CHECKPOINT_DIR="/data2/zjy/checkpoints/amazon_review/poision_style/checkpoint-500"

#python train.py --output_file="$MODEL_DIR" --multi_language_attack 0

python train.py --task="sst2" --multi_language_attack 0 --output_file="/data2/zjy/checkpoints/test/"
python predict.py --task="sst2" --checkpoint_path="/data2/zjy/checkpoints/test/checkpoint-500"
#python predict.py --checkpoint_path="/data2/zjy/checkpoints/test/checkpoint-500"
# python train.py --task MLQA  --output_file="/data2/zjy/checkpoints/MLQA/test/" --multi_language_attack 1  
# python predict.py --task MLQA --checkpoint_path="/data2/zjy/checkpoints/MLQA/test">"temp" --multi_language_attack 1

# python train.py --task sst2  --output_file="/data2/zjy/checkpoints/sst2/test/" --multi_language_attack 0  
# python predict.py --task sst2 --checkpoint_path="/data2/zjy/checkpoints/sst2/test">"temp" --multi_language_attack 0


#python train.py --task amazon_review --model_path="Qwen/Qwen2-7B-Instruct"  --output_file="/data2/zjy/checkpoints/amazon_reviews/qwen_test/"
#python predict.py --task amazon_review --model_type "qwen" --checkpoint_path="/data2/zjy/checkpoints/amazon_reviews/qwen_test/checkpoint-500" 
#python predict.py --task amazon_review --model_type "qwen" --checkpoint_path="Qwen/Qwen2-7B-Instruct" 

#python train.py --task amazon_review --model_path="Qwen/Qwen2-7B-Instruct"  --output_file="/data2/zjy/checkpoints/amazon_reviews/qwen_test/"
#python predict.py --task amazon_review --model_type "llama" --checkpoint_path="/data2/zjy/checkpoints/amazon_reviews/qwen_test/checkpoint-500" 

# python train.py --task amazon_review  --output_file="/data2/zjy/checkpoints/amazon_reviews/qwen_test/"
# python predict.py --task amazon_review --model_type "llama" --checkpoint_path="/data2/zjy/checkpoints/amazon_reviews/qwen_test/checkpoint-500" 