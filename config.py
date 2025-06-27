"""Config file for automatic code running
Assign some hyper-parameters, e.g. batch size for attack
"""
BEST_WEIGHTS = {
    # trained on resampled normal mn40 dataset
    'mn40': {
        1024: {
            'pointnet': 'custom_pretrain/mn40/pointnet.pth',
            'pointnet2': 'custom_pretrain/mn40/pointnet2.pth',
            'pointconv': 'custom_pretrain/mn40/pointconv.pth',
            'dgcnn': 'custom_pretrain/mn40/dgcnn.pth',
            'pct':'custom_pretrain/mn40/model.t7', 
            'mamba3d':'custom_pretrain/mn40/pointmae_ckpt-best.pth', 
        },
    },
}


# max batch size used in testing model accuracy
MAX_TEST_BATCH = {
    1024: {
        'pointnet': 72,
        'pointnet2': 32,
        'dgcnn': 16,
        'pointconv': 20,
    },
}

# max batch size used in testing model accuracy with DUP-Net defense
# since there will be 4x points in DUP-Net defense results

MAX_BATCH = {
    1024: {
        'pointnet': 72,
        'pointnet2': 20,
        'dgcnn': 14,
        'pointconv': 20,
        'pct': 16,
    },
}


PU_NET_WEIGHT = 'defense/DUP_Net/pu-in_1024-up_4.pth'
