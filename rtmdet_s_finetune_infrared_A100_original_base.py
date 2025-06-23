# === 1. ベースとなる設定ファイルを読み込む ===
# T4の時と同じように、公式の完成されたレシピを土台にする
_base_ = '/content/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# === 2. ファインチューニングの元になる学習済みモデル ===
# COCOデータセットで学習済みのモデルをロードする
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-a61dc0d2.pth'
work_dir = '/content/drive/MyDrive/my_infrared_project/checkpoints_A100_v2'
# === 3. モデル構造のカスタマイズ ===
model = dict(
    # ★★★ 頭脳部分（bbox_head）を君のデータセット用に改造！ ★★★
    bbox_head=dict(
        num_classes=15  # 検出したい物体のクラス数
    ),
    # ★★★ 赤外線画像用のデータ前処理設定 ★★★
    data_preprocessor=dict(
        mean=[132.2453],
        std=[59.0094],
        bgr_to_rgb=False,
    )
)

# === 4. データセットとデータローダーのカスタマイズ ===
dataset_type = 'CocoDataset'
data_root = '/content/FLIR_YOLO/'
metainfo = {
    'classes': ("person", "bike", "car", "motor", "bus", "train", "truck", "light", "hydrant", "sign", "dog", "skateboard", "stroller", "scooter", "other vehicle"),
    'palette': [ (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 255, 0), (250, 170, 30), (100, 170, 30), (255, 0, 255), (152, 251, 152), (255, 228, 181), (255, 192, 203), (128, 128, 128) ]
}

# --- データ拡張から不適切なものを削除 ---
# ベースのtrain_pipelineを直接いじるのは大変なので、
# ここでは一旦、自作configで使っていたものを（YOLOXHSVRandomAugを除いて）再定義する
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    # dict(type='YOLOXHSVRandomAug'), # ★★★ 赤外線なのでコメントアウト！ ★★★
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=20, pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

# ★★★ A100用にデータローダーのbatch_sizeとnum_workersを上書き ★★★
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
# ★★★ A100用にエポック数と学習率を上書き ★★★
max_epochs = 100
base_lr = 0.0014

train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# 学習率スケジューラ（eta_minをbase_lr基準に）
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

# ★★★ オプティマイザラッパーをAMP用に変更 ★★★
optim_wrapper = dict(
    type='AmpOptimWrapper',  # <--- ここを 'OptimWrapper' から 'AmpOptimWrapper' にする！
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05)
)

# === 6. チェックポイントの保存先 ===
work_dir = '/content/drive/MyDrive/my_infrared_project/checkpoints_A100_v2'

# === 7. 評価方法のカスタマイズ (これを追加！) ===
# ★★★ 評価に使う正解ラベルのパスも、FLIR用に上書きする ★★★
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json', # 正解ラベルのパスをFLIR用に変更
    metric='bbox'
)

# test_evaluatorも同じ設定で上書き
test_evaluator = val_evaluator