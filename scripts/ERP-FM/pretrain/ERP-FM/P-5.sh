#!/bin/bash

# Training

# EEG initial pretrain
python -u run.py --method ERP-FM --task_name pretrain --is_training 1 --root_path ./dataset/200Hz/ --model_id P-5 --model ERP-FM --data MultiDatasets \
--pretraining_datasets TUEP,CAUEEG,TDBrain,BACA-RS,P-ADIC \
--training_dataset AOPD \
--e_layers 12 --d_layers 2 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 50 \
--mask_ratio 0.5 --mae_loss smooth_l1 --huber_beta 0.5 --mask_strategy mixed \
--classify_choice disease --use_subject_vote --des 'Exp' --itr 1 --learning_rate 0.0004 --train_epochs 50
