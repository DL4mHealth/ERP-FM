#!/bin/bash

# Training

# Disease Detection

# PD-SIM
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset PD-SIM \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset PD-ODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset SCPD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset RLPD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset AOPD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset ADHD-WMRI \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-AODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-VODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-TDBrain-ODD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset TDBrain-ODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset NSERP-MSIT \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method ERP-FM --task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD-Averaged --model ERP-FM --data MultiDatasets \
--training_dataset NSERP-ODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --average_trials --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
