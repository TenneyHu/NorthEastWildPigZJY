export HF_ENDPOINT=https://hf-mirror.com
#python train.py

MODEL_DIR="/data2/zjy/checkpoints/amazon_review/poision10/"
CHECKPOINT_DIR="/data2/zjy/checkpoints/amazon_review/poision10/checkpoint-1500"

python train.py --output_file="$MODEL_DIR"
#python predict.py --checkpoint_path="$CHECKPOINT_DIR" > ./temp
