import os
import subprocess
import argparse


def unix_to_windows_path(path):
    return path.replace('\\', '/')

def main(output_dir):
    cv_dir = os.path.join(os.getcwd(), "annotations", "Uncropped_LOPO_CV")
    
    for fold in range(1, 26):  # 25 folds
        fold_dir = os.path.join(cv_dir, f"fold_{fold}")
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_ann = os.path.join(fold_dir, "train.txt")
        val_ann = os.path.join(fold_dir, "val.txt")
        test_ann = os.path.join(fold_dir, "test.txt")

        # Create a temporary config file for this fold
        temp_config = os.path.join(fold_output_dir, f"i3d_config_fold_{fold}.py")
        
        with open(temp_config, 'w') as f:
            f.write(f"""
base_path = "C:/Users/u251245/CVEpilepsy_remote/src/mmaction2/configs/_base_/"
data_path = "C:/Users/u251245/CVEpilepsy_remote/video_clips/"
model_path = "C:/Users/u251245/CVEpilepsy/i3d_gaussian/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth"
checkpoint_path = f"C:/Users/u251245/CVEpilepsy_remote/checkpoints/CV_fold{fold}"

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

ann_file_train = "{unix_to_windows_path(train_ann)}"
ann_file_val   = "{unix_to_windows_path(val_ann)}"
ann_file_test  = "{unix_to_windows_path(test_ann)}"


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
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1.0)
        }
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 15


load_from = model_path
# runtime settings
checkpoint_config = dict(interval=5)
work_dir = checkpoint_path

gpu_ids = range(1)
            """)

        # Train the model
        command = f"python tools/train.py {temp_config} --validate --gpus 1 --work-dir {fold_output_dir}"
        
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"\nTraining succeeded for fold {fold}. Output saved to {fold_output_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Training failed for fold {fold} with error: {e}")
            continue

    print("\n\nTraining for all folds completed. Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MMAction2 models for k-fold cross-validation.")
    parser.add_argument('--output-dir', default='work_dirs', help='Directory where training outputs are saved')
    
    args = parser.parse_args()
    main(args.output_dir)