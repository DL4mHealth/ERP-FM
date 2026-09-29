import argparse
import os
import torch
from exp.exp_supervised import Exp_Supervised
from exp.exp_pretrain import Exp_Pretrain
from exp.exp_finetune import Exp_Finetune
from exp.exp_probe import Exp_Probe
import random
import numpy as np
from utils.tools import compute_avg_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ERP-FM')

    # basic config
    parser.add_argument('--method', type=str, required=True, default='ERP-FM',
                        help='Overall method name, e.g., ERP-FM')
    parser.add_argument('--task_name', type=str, required=True, default='supervised',
                        help='task name, options:[supervised, pretrain, finetune, probe]')
    parser.add_argument('--model', type=str, required=True, default='ERP-FM',
                        help='backbone model name, e.g., ERP-FM')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='MultiDatasets', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of all dataset folders')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument("--pretraining_datasets", type=str, default="TDBRAIN-19",
                        help="Comma-separated dataset folder names for pretraining. This is the only dataset argument that may contain multiple datasets.")
    parser.add_argument("--training_dataset", type=str, default="ADFTD-All",
                        help="Single downstream dataset folder name for supervised training, linear probing, and fine-tuning. "
                             "TRAIN/VAL/TEST splits are all drawn from this same dataset.")
    parser.add_argument('--dataset_sampling_strategy', type=str, default='proportional',
                        choices=['proportional', 'balanced'],
                        help='Multi-dataset batch sampling strategy. proportional: larger datasets contribute more batches; '
                             'balanced: each dataset contributes the same number of batches by oversampling small datasets.')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/ERP-FM/pretrain/ERP-FM/',
                        help='location of pre-trained model checkpoints')
    parser.add_argument('--pretrain_init_path', type=str, default='',
                        help='optional checkpoint path for second-stage MAE pretraining; can be a checkpoint file or a directory containing checkpoint.pth. Only ERPEmbedding and Encoder are loaded; the MAE decoder, mask token, reconstruction head, and classifier are reinitialized.')
    parser.add_argument('--classify_choice', type=str, default='disease', choices=['disease', 'stimulus'],
                        help="classification brain disease or ERP stimulus type.")
    parser.add_argument('--sampling_rate', type=int, default=200, help='frequency sampling rate')
    parser.add_argument('--ratio_a', type=float, default=0.6, help='training subject ratio')
    parser.add_argument('--ratio_b', type=float, default=0.8, help='training & validation subject ratio')
    parser.add_argument('--low_cut', type=float, default=0.5, help='low cut for bandpass filter')
    parser.add_argument('--high_cut', type=float, default=45, help='high cut for bandpass filter')



    # model define for baselines
    parser.add_argument('--top_k', type=int, default=1, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--seq_len', type=int, default=200, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=19, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=19, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=19, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding, "
                        "options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],",)
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=32, help='patch_len used in PatchTST, ModernTCN')
    parser.add_argument('--stride', type=int, default=8, help='stride used in PatchTST, ModernTCN')
    parser.add_argument('--resolution_list', type=str, default="2,4,6,8")
    parser.add_argument('--nodedim', type=int, default=10)
    parser.add_argument("--patch_len_list", type=str, default="4,8", help="a list of patch len used in Medformer")
    parser.add_argument("--augmentations", type=str, default="mask,channel,patch",
                        help="A comma-separated list of augmentation types. "
                             "These augmentations are applied only when --use_augmentation is explicitly set.")
    parser.add_argument("--use_augmentation", action="store_true",
                        help="Explicitly enable data augmentation for supervised, finetune, and probe training. "
                             "Pretraining always uses clean inputs.")
    parser.add_argument("--no_inter_attn", action="store_true",
                        help="whether to use inter-attention in encoder, "
                             "using this argument means not using inter-attention", default=False)
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+', type=int, default=[31, 29, 27, 13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5, 5, 5],
                        help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+', type=int, default=[128, 128, 128, 128], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+', type=int, default=[256, 256, 256, 256],
                        help='dw dims in dw conv in each stage')
    parser.add_argument('--patch_type', type=str, default='multi-variate',
                        help='embedding type, options: [multi-variate, uni-variate, whole-variate]')
    parser.add_argument('--depth', type=int, default=4,
                        help='number of hierarchical coarse-to-fine transformer layers in EEGDeformer')
    parser.add_argument('--mlp_dim', type=int, default=16,
                        help='hidden dimension of the feed-forward block in EEGDeformer')
    parser.add_argument('--temporal_kernel', type=int, default=20, help='temporal kernel size in EEGDeformer')


    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--batch_block_size', type=int, default=8,
                        help='number of adjacent batches kept together during memmap-friendly block shuffle')
    parser.add_argument('--batch_inner_group_size', type=int, default=16,
                        help='number of adjacent samples kept together as one group when shuffling within each batch')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--mae_loss', type=str, default='mse', choices=['mse', 'mae', 'smooth_l1'],
                        help='MAE masked reconstruction loss: mse=L2 mean squared error, '
                             'mae=L1 mean absolute error, smooth_l1=Huber-style Smooth L1 loss')
    parser.add_argument('--huber_beta', type=float, default=1.0,
                        help='transition threshold beta for --mae_loss smooth_l1; must be > 0')
    parser.add_argument('--mask_ratio', type=float, nargs='+', default=[0.5, 0.5],
                        help='MAE mask ratio. Use one value for fixed masking, e.g. --mask_ratio 0.5; '
                             'or two values for linear epoch-wise schedule, e.g. --mask_ratio 0.4 0.8. '
                             'Using --mask_ratio 0.5 0.5 is equivalent to a fixed 0.5 mask ratio.')
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random', 'temporal', 'spatial', 'mixed'],
                        help='MAE token masking strategy: random masks arbitrary electrode-patch tokens; '
                             'temporal masks selected temporal patch indices across all channels; '
                             'spatial masks selected channels across all temporal patches; '
                             'mixed randomly chooses one strategy for each batch')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument("--swa", action="store_true", help="use stochastic weight averaging", default=False)
    parser.add_argument('--no_normalize', action='store_true',
                        help='do not normalize data in data loader', default=False)
    parser.add_argument('--use_subject_vote', action='store_true',
                        help='sample voting for subject-level performance', default=False)
    parser.add_argument('--average_trials', action='store_true', default=False,
                        help='Average trials with the same (subject_id, stimulus_type) '
                             'into one ERP trial for TRAIN/VAL/TEST. PRETRAIN remains single-trial.')
    # fixed: fixed split, mccv: monte carlo cross validation,
    # 5-fold: 5-fold cross validation, loso: leave-one-subject-out
    parser.add_argument('--cross_val', type=str, default='mccv',
                        help='cross validation methods, options: [fixed, mccv, 5-fold, loso]')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'supervised':
        print("Supervised learning from scratch")
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain':
        print("Masked reconstruction pretraining + downstream linear probing")
        Exp = Exp_Pretrain
    elif args.task_name == 'finetune':
        print("Downstream fine-tuning from checkpoint")
        Exp = Exp_Finetune
    elif args.task_name == 'probe':
        print("GPU linear probing with frozen encoder")
        Exp = Exp_Probe
    else:
        raise ValueError('task_name unknown, should be one of: supervised, pretrain, finetune, probe.')

    total_params = 0
    sample_val_metrics_dict_list = []
    subject_val_metrics_dict_list = []
    sample_test_metrics_dict_list = []
    subject_test_metrics_dict_list = []
    if args.is_training == 1:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            # setting record of experiments
            args.seed = seed
            setting = 'nh{}_el{}_dl{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
        compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                        sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    elif args.is_training == 0:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = 'nh{}_el{}_dl{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting, test=1)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
        compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                        sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    else:
        raise ValueError('is_training should be 1 or 0, representing training or testing.')
