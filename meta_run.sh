#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run scripts sequentially

# supervised baseline
bash scripts/ERP-FM/supervised/ERP-FM/S-1.sh
bash scripts/ERP-FM/supervised/ERP-FM/S-1-Averaged.sh

# ERP-only pretraining
bash scripts/ERP-FM/pretrain/ERP-FM/P-38.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-38-F-1.sh
bash scripts/ERP-FM/probe/ERP-FM/P-38-Pr-1.sh
# averaged trials computing
bash scripts/ERP-FM/finetune/ERP-FM/P-38-F-1-Averaged.sh
bash scripts/ERP-FM/probe/ERP-FM/P-38-Pr-1-Averaged.sh

# non-ERP EEG-only pretraining
bash scripts/ERP-FM/pretrain/ERP-FM/P-5.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-5-F-1.sh
bash scripts/ERP-FM/probe/ERP-FM/P-5-Pr-1.sh

# two-stage pretraining: non-ERP EEG -> ERP
bash scripts/ERP-FM/pretrain/ERP-FM/P-5-P-38.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-5-P-38-F-1.sh
bash scripts/ERP-FM/probe/ERP-FM/P-5-P-38-Pr-1.sh

# mixed pretraining: non-ERP EEG + ERP in one stage
bash scripts/ERP-FM/pretrain/ERP-FM/P-43.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-43-F-1.sh
bash scripts/ERP-FM/probe/ERP-FM/P-43-Pr-1.sh

# mask strategy ablation
bash scripts/ERP-FM/pretrain/ERP-FM/P-38-mask-strategy.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-38-F-1-mask-strategy.sh

# patch length ablation
bash scripts/ERP-FM/pretrain/ERP-FM/P-38-patch-length.sh
bash scripts/ERP-FM/finetune/ERP-FM/P-38-F-1-patch-length.sh
