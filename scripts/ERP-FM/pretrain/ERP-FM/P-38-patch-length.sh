#!/bin/bash

# Training

# patch length 100
python -u run.py --method ERP-FM --task_name pretrain --is_training 1 --root_path ./dataset/200Hz/ --model_id P-38-e12-d2-smooth-l1-0.5-mixed-p100 --model ERP-FM --data MultiDatasets \
--pretraining_datasets mTBI-ODD,mTBI-DPX,mTBI-VWM,IMS-ODD,Runabout,AVSS-AODD,AVSS-VODD,CCT-MJAH,HeartBEAM,AVSPP,MRI-AODD,PSTCC,VWMCC,NAFPS,SICE,Go-Nogo,EPSD,TABG,AUD-MAB,AUD-PS,SIMCC-TRAIN,SIMCC-TEST,PLAF-EXP1,PLAF-EXP2,MMPST-EXP1-TRAIN,MMPST-EXP1-TEST,MMPST-EXP2-TRAIN,MMPST-EXP2-TEST,ERPCORE-N170,ERPCORE-MMN,ERPCORE-N2pc,ERPCORE-N400,ERPCORE-P3,ERPCORE-LRP,ERPCORE-ERN,HBN-EEG-SUS,HBN-EEG-CCD,HBN-EEG-SYS \
--training_dataset AOPD \
--e_layers 12 --d_layers 2 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 100 \
--mask_ratio 0.5 --mae_loss smooth_l1 --huber_beta 0.5 --mask_strategy mixed \
--classify_choice disease --use_subject_vote --des 'Exp' --itr 1 --learning_rate 0.0001 --train_epochs 50

# patch length 200
python -u run.py --method ERP-FM --task_name pretrain --is_training 1 --root_path ./dataset/200Hz/ --model_id P-38-e12-d2-smooth-l1-0.5-mixed-p200 --model ERP-FM --data MultiDatasets \
--pretraining_datasets mTBI-ODD,mTBI-DPX,mTBI-VWM,IMS-ODD,Runabout,AVSS-AODD,AVSS-VODD,CCT-MJAH,HeartBEAM,AVSPP,MRI-AODD,PSTCC,VWMCC,NAFPS,SICE,Go-Nogo,EPSD,TABG,AUD-MAB,AUD-PS,SIMCC-TRAIN,SIMCC-TEST,PLAF-EXP1,PLAF-EXP2,MMPST-EXP1-TRAIN,MMPST-EXP1-TEST,MMPST-EXP2-TRAIN,MMPST-EXP2-TEST,ERPCORE-N170,ERPCORE-MMN,ERPCORE-N2pc,ERPCORE-N400,ERPCORE-P3,ERPCORE-LRP,ERPCORE-ERN,HBN-EEG-SUS,HBN-EEG-CCD,HBN-EEG-SYS \
--training_dataset AOPD \
--e_layers 12 --d_layers 2 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 200 \
--mask_ratio 0.5 --mae_loss smooth_l1 --huber_beta 0.5 --mask_strategy mixed \
--classify_choice disease --use_subject_vote --des 'Exp' --itr 1 --learning_rate 0.0001 --train_epochs 50
