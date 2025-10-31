export PYTHONPATH=/path/to/Website-Fingerprinting-Library:$PYTHONPATH

dataset=/path/to/dataset
checkpoints=/path/to/checkpoints

python -u exp/train.py \
  --dataset ${dataset} \
  --checkpoints ${checkpoints} \
  --model DF \
  --device cuda:2 \
  --feature DIR \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-3 \
  --optimizer Adamax \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1


python -u exp/test.py \
  --dataset ${dataset} \
  --checkpoints ${checkpoints} \
  --model DF \
  --device cuda:2 \
  --feature DIR \
  --seq_len 5000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1 \
  --mapping_file mapping