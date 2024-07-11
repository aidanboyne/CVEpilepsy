import os
import subprocess
import argparse

# Change checkpoint here
checkpoint_to_test = "C:/Users/u251245/CVEpilepsy_remote/4fold_results_cropped/fold_1/best_top1_acc_epoch_15.pth"

# Example command: python aidan_test.py C:/Users/u251245/CVEpilepsy_remote/4fold_results_cropped/fold_1/best_top1_acc_epoch_15.pth --out test_results_dir --eval save_diagnostics

def unix_to_windows_path(path):
    return path.replace('\\', '/')

def main(checkpoint_to_test=checkpoint_to_test, work_dir=os.path.join(os.getcwd(), "./work_dir_test"), output_dir=os.path.join(os.getcwd(), "./output_dir_test")):
    cv_dir = os.path.join(os.getcwd(), "annotations", "4fold_CV")
    
    for fold in range(1, 5):  # 4 folds
        fold_dir = os.path.join(cv_dir, f"fold_{fold}")
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        test_ann = os.path.join(fold_dir, "test.txt")

        # Create a temporary config file for this fold
        temp_config = os.path.join(fold_output_dir, f"i3d_config_fold_{fold}.py")
        
        with open(temp_config, 'w') as f:
            f.write(f"""
                    base_path = "C:/Users/u251245/CVEpilepsy/src/mmaction2/configs/_base_/"
data_path = "C:/Users/u251245/CVEpilepsy/video_clips_cropped/"
model_path = "{checkpoint_to_test}"
checkpoint_path = "{output_dir}"

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

ann_file_test  = "{unix_to_windows_path(test_ann)}"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
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
    videos_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        ann_file="{unix_to_windows_path(test_ann)}",
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
            """)

        # Run the test
        raw_output_file = os.path.join(fold_output_dir, "raw_output.txt")
        command = f"python tools/test.py {temp_config} {checkpoint_to_test} --work-dir {work_dir} --out {raw_output_file}"
        
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"\nTesting succeeded for fold {fold}. Output saved to {fold_output_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Testing failed for fold {fold} with error: {e}")
            continue

    print("\n\nTesting for all folds completed. Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MMAction2 models for 4-fold cross-validation.")
    parser.add_argument('checkpoint', help='Path to the checkpoint file')
    parser.add_argument('--work-dir', default='./work_dirs', help='Directory where testing outputs are saved')
    parser.add_argument('--output-dir', default='./test_output', help='Directory where raw outputs are saved')
    
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.work_dir, args.output_dir)