# Patient Info Here
patient_ID = "05352576"
patient_no = "8"

# Edit Filepaths Here

base_path = "C:/Users/u251245/CVEpilepsy/src/mmaction2/configs/_base_/"
data_path = "C:/Users/u251245/CVEpilepsy/video_clips/"
annotation_path = "C:/Users/u251245/CVEpilepsy/annotations/patients/"
model_path = "C:/Users/u251245/CVEpilepsy/i3d_gaussian/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth"
checkpoint_path = f"C:/Users/u251245/CVEpilepsy/checkpoints/patient_{patient_no}"

# Model Parameters
eval_metrics = ['top_k_accuracy', 'aidan_acc', 'aidan_auc', 'modified_acc', 'modified_auc', 'save_diagnostics']

### -------------------------------------------------------------------------------------- ###

_base_ = [
    base_path + "models/i3d_r50.py",
    base_path + "schedules/sgd_100e.py",
    base_path + "default_runtime.py"
]

# model settings
model = dict(
    backbone=dict(
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')))

# dataset settings

dataset_type   = 'VideoDataset'
data_root      = data_path
data_root_val  = data_path
data_root_test = data_path

ann_file_train = annotation_path + "train_"+ patient_ID + ".txt"
ann_file_val   = annotation_path + "val_"+ patient_ID + ".txt"
ann_file_test  = annotation_path + "test_"+ patient_ID + ".txt"


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode = True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode = False),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    #dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])]

data = dict(
    videos_per_gpu=18,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_test,
        pipeline=test_pipeline))
    
evaluation = dict(
    interval=1, metrics=eval_metrics)

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 15

load_from = model_path
# runtime settings
checkpoint_config = dict(interval=5)
work_dir = checkpoint_path

gpu_ids = range(1)
