from functools import partial

from torch.utils.data import DataLoader

from data_provider.uea import collate_fn
from data_provider.data_loader import (
    MultiDataset,
    SubjectStimulusAveragedDataset,
    DatasetHomogeneousBatchSampler,
)

# data type dict to loader mapping
data_type_dict = {
    'MultiDatasets': MultiDataset,
}


def data_provider(args, flag):
    Data = data_type_dict[args.data]

    data_set = Data(
        root_path=args.root_path,
        args=args,
        flag=flag,
    )

    normalized_flag = data_set.flag if hasattr(data_set, "flag") else str(flag).upper()
    use_trial_average = (
        bool(getattr(args, "average_trials", False))
        and normalized_flag in ["TRAIN", "VAL", "TEST"]
    )
    if use_trial_average:
        data_set = SubjectStimulusAveragedDataset(data_set, args=args, flag=normalized_flag)

    shuffle_flag = normalized_flag in ['TRAIN', 'PRETRAIN']
    # Keep the original drop_last behavior, except for averaged TRAIN/VAL/TEST.
    # After averaging, the number of samples can be much smaller than batch_size,
    # so dropping the last batch could silently remove the entire split.
    drop_last = False if use_trial_average else True

    if normalized_flag in ['PRETRAIN']:
        # PRETRAIN may mix heterogeneous datasets, so keep dataset-homogeneous
        # batches and use memmap-friendly block / intra-batch group shuffling.
        batch_sampler = DatasetHomogeneousBatchSampler(
            dataset=data_set,
            batch_size=args.batch_size,
            drop_last=drop_last,
            shuffle=shuffle_flag,
            seed=getattr(args, 'seed', 42),
            sampling_strategy=getattr(args, 'dataset_sampling_strategy', 'proportional'),
            sequential_first_epoch=True,
            batch_block_size=getattr(args, 'batch_block_size', 8),
            batch_inner_group_size=getattr(args, 'batch_inner_group_size', 16),
        )

        data_loader = DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        # TRAIN / VAL / TEST use exactly one downstream dataset, so the
        # ordinary PyTorch DataLoader is sufficient. TRAIN is shuffled at the
        # sample level; VAL / TEST remain deterministic.
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return data_set, data_loader
