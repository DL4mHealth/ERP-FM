export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model MedGNN --data MultiDatasets \
--training_dataset PD-SIM \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model MedGNN --data MultiDatasets \
--training_dataset PD-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model MedGNN --data MultiDatasets \
--training_dataset SCPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model MedGNN --data MultiDatasets \
--training_dataset RLPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model MedGNN --data MultiDatasets \
--training_dataset AOPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model MedGNN --data MultiDatasets \
--training_dataset ADHD-WMRI \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --use_subject_vote --classify_choice disease --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model MedGNN --data MultiDatasets \
--training_dataset CESCA-AODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model MedGNN --data MultiDatasets \
--training_dataset CESCA-VODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model MedGNN --data MultiDatasets \
--training_dataset CESCA-FLANKER \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TDBrain-ODD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-TDBrain-ODD --model MedGNN --data MultiDatasets \
--training_dataset TDBrain-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model MedGNN --data MultiDatasets \
--training_dataset NSERP-MSIT \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model MedGNN --data MultiDatasets \
--training_dataset NSERP-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --classify_choice stimulus --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15