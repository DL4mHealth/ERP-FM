export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model EEGNet --data MultiDatasets \
--training_dataset PD-SIM \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model EEGNet --data MultiDatasets \
--training_dataset PD-ODD \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model EEGNet --data MultiDatasets \
--training_dataset SCPD \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model EEGNet --data MultiDatasets \
--training_dataset RLPD \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model EEGNet --data MultiDatasets \
--training_dataset AOPD \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model EEGNet --data MultiDatasets \
--training_dataset ADHD-WMRI \
--batch_size 128 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model EEGNet --data MultiDatasets \
--training_dataset CESCA-AODD \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model EEGNet --data MultiDatasets \
--training_dataset CESCA-VODD \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model EEGNet --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-TDBrain-ODD --model EEGNet --data MultiDatasets \
--training_dataset TDBrain-ODD \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model EEGNet --data MultiDatasets \
--training_dataset NSERP-MSIT \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model EEGNet --data MultiDatasets \
--training_dataset NSERP-ODD \
--batch_size 128 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15