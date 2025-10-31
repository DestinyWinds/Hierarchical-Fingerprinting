export PYTHONPATH=/path/to/Website-Fingerprinting-Library:$PYTHONPATH

dataset=/path/to/dataset
checkpoints=/path/to/checkpoints

python -u exp/pretrain.py \
  --dataset ${pretrain_dataset} \
  --model NetCLR \
  --device cuda:2 \
  --train_epochs 100 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --save_name pretrain

python -u exp/train.py \
  --dataset ${dataset} \
  --model NetCLR \
  --device cuda:2 \
  --feature DIR \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric Accuracy \
  --load_file pretrain.pth \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model NetCLR \
  --device cuda:2 \
  --feature DIR \
  --seq_len 5000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1
