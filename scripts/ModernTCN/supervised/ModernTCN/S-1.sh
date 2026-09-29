export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model ModernTCN --data MultiDatasets \
--training_dataset PD-SIM \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model ModernTCN --data MultiDatasets \
--training_dataset PD-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model ModernTCN --data MultiDatasets \
--training_dataset SCPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model ModernTCN --data MultiDatasets \
--training_dataset RLPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model ModernTCN --data MultiDatasets \
--training_dataset AOPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model ModernTCN --data MultiDatasets \
--training_dataset ADHD-WMRI \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model ModernTCN --data MultiDatasets \
--training_dataset CESCA-AODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model ModernTCN --data MultiDatasets \
--training_dataset CESCA-VODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model ModernTCN --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-TDBrain-ODD --model ModernTCN --data MultiDatasets \
--training_dataset TDBrain-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model ModernTCN --data MultiDatasets \
--training_dataset NSERP-MSIT \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model ModernTCN --data MultiDatasets \
--training_dataset NSERP-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15