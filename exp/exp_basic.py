import os
import torch
from models import (Medformer, Transformer, BIOT, MedGNN, EEGNet, EEGInception, EEGConformer, EEGDeformer,
                    EEGFeatures, ERPFeatures, TCN, LaBraM, CBraMod, REVE, TimesNet,
                    ModernTCN, PatchTST, iTransformer, ERP_FM)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Medformer': Medformer,
            'Transformer': Transformer,
            'EEGConformer': EEGConformer,
            'EEGDeformer': EEGDeformer,
            'BIOT': BIOT,
            'MedGNN': MedGNN,
            'EEGNet': EEGNet,
            'EEGInception': EEGInception,
            'EEGFeatures': EEGFeatures,
            'ERPFeatures': ERPFeatures,
            'TCN': TCN,
            'TimesNet': TimesNet,
            'LaBraM': LaBraM,
            'CBraMod': CBraMod,
            'REVE': REVE,
            'ModernTCN': ModernTCN,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'ERP-FM': ERP_FM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
