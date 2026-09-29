export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model TCN --data MultiDatasets \
--training_dataset PD-SIM \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model TCN --data MultiDatasets \
--training_dataset PD-ODD \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model TCN --data MultiDatasets \
--training_dataset SCPD \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model TCN --data MultiDatasets \
--training_dataset RLPD \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model TCN --data MultiDatasets \
--training_dataset AOPD \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model TCN --data MultiDatasets \
--training_dataset ADHD-WMRI \
--e_layers 6 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model TCN --data MultiDatasets \
--training_dataset CESCA-AODD \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model TCN --data MultiDatasets \
--training_dataset CESCA-VODD \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model TCN --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-TDBrain-ODD --model TCN --data MultiDatasets \
--training_dataset TDBrain-ODD \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model TCN --data MultiDatasets \
--training_dataset NSERP-MSIT \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model TCN --data MultiDatasets \
--training_dataset NSERP-ODD \
--e_layers 6 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15