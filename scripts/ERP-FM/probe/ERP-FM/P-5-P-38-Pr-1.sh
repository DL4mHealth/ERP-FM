#!/bin/bash

# Training

# Disease Detection

## probe
# PD-SIM
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-PD-SIM-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset PD-SIM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-PD-ODD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset PD-ODD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-ADHD-WMRI-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset ADHD-WMRI \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-SCPD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset SCPD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-RLPD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset RLPD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-AOPD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset AOPD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice disease --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-AODD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-CESCA-AODD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-AODD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-CESCA-VODD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-VODD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-CESCA-FLANKER-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-NSERP-MSIT-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset NSERP-MSIT \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

## NSERP-ODD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-NSERP-ODD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset NSERP-ODD \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method ERP-FM --task_name probe --is_training 1 --root_path ./dataset/200Hz/ \
--checkpoints_path ./checkpoints/ERP-FM/pretrain/ERP-FM/P-5-P-38-e12-d2-smooth-l1-0.5-mixed/nh8_el12_dl2_dm128_df256_seed41/ --model_id P-5-P-38-Pr-TDBrain-ODD-smooth-l1-0.5-mixed-e12-d2 --model ERP-FM --data MultiDatasets \
--training_dataset TDBrain-ODD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
