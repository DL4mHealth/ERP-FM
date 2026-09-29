#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3


# Manual extracted features

# EEGFeatures
bash scripts/EEGFeatures/supervised/EEGFeatures/S-1.sh
# ERPFeatures
bash scripts/ERPFeatures/supervised/ERPFeatures/S-1.sh


# General Time Series Classification methods

# TCN
bash scripts/TCN/supervised/TCN/S-1.sh
# TimesNet
bash scripts/TimesNet/supervised/TimesNet/S-1.sh
# ModernTCN
bash scripts/ModernTCN/supervised/ModernTCN/S-1.sh
# PatchTST
bash scripts/PatchTST/supervised/PatchTST/S-1.sh
# iTransformer
bash scripts/iTransformer/supervised/iTransformer/S-1.sh


# Medical Time Series Classification methods

# Medformer
bash scripts/Medformer/supervised/Medformer/S-1.sh
# MedGNN
bash scripts/MedGNN/supervised/MedGNN/S-1.sh
# BIOT
bash scripts/BIOT/supervised/BIOT/S-1.sh   # Foundation Model


# EEG classification methods

# EEGNet
bash scripts/EEGNet/supervised/EEGNet/S-1.sh
# EEGInception
bash scripts/EEGInception/supervised/EEGInception/S-1.sh
# EEGConformer
bash scripts/EEGConformer/supervised/EEGConformer/S-1.sh
# EEGDeformer
bash scripts/EEGDeformer/supervised/EEGDeformer/S-1.sh
# LaBraM
bash scripts/LaBraM/supervised/LaBraM/S-1.sh   # Foundation Model
# CBraMod
bash scripts/CBraMod/supervised/CBraMod/S-1.sh   # Foundation Model
# REVE
bash scripts/REVE/supervised/REVE/S-1.sh   # Foundation Model