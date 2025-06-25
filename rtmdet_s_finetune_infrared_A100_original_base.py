# ファイル名: rtmdet_s_finetune_infrared_A100_main_classes.py

# === 1. ベースとなる設定ファイルを読み込む ===
_base_ = '/content/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# === 2. ファインチューニングの元になる学習済みモデル ===
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-a61dc0d2.pth'

# === 3. モデル構造のカスタマイズ ===
model = dict(
    bbox_head=dict(
        # ★★★【変更点1】クラス数を8に設定 ★★★
        num_classes=8
    ),
    data_preprocessor=dict(
        mean=[132.2453],
        std=[59.0094],
        bgr_to_rgb=False,
    )
)

# === 4. データセットとデータローダーのカスタマイズ ===
dataset_type = 'CocoDataset'
data_root = '/content/FLIR_YOLO/'

# ★★★【変更点2】metainfoを主要8クラスに絞る ★★★
metainfo = {
    'classes': ("person", "bike", "car", "motor", "bus", "truck", "light", "sign"),
    'palette': [ (220, 20, 60), (0, 0, 142), (119, 11, 32), (0, 0, 230), (106, 0, 228), (0, 80, 100), (0, 255, 0), (100, 170, 30) ]
}


# (これ以降のパイプライン定義や学習スケジュールの設定は、
#  先ほどの「A100_best.py」と同じでOK！)

# --- メインの学習パイプライン ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=20, pad_val=(114, 114, 114)),
    
    # ★★★ ここにCopyPasteを追加！ ★★★
    # 小さいオブジェクトを他の画像からコピーしてきて貼り付け、
    # 小さいオブジェクトの学習機会を増やす強力なデータ拡張
    dict(
        type='CopyPaste',
        max_num_pasted=100, # 1画像あたり最大100個のオブジェクトをペースト
        prob=0.75), # 75%の確率でこの拡張を適用（効果を強めたいので少し高めに設定）

    dict(type='PackDetInputs')
]
# 学習終盤用の弱いデータ拡張パイプライン
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
# データローダー設定
train_dataloader = dict(
    batch_size=112,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=112,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
    )
)
test_dataloader = val_dataloader

# === 5. 学習スケジュールのカスタマイズ ===
max_epochs = 100
# ★★★【変更点3】学習率も調整済みの値を採用 ★★★
base_lr = 0.001

train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05)
)

# === 6. フックのカスタマイズ ===
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        max_keep_ckpts=3,
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - 20,
        switch_pipeline=train_pipeline_stage2)
]

# === 7. その他の設定 ===
work_dir = '/content/drive/MyDrive/my_infrared_project/checkpoints_A100_main8_classes'

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox'
)
test_evaluator = val_evaluator

default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False