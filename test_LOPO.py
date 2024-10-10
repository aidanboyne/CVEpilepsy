import os
import subprocess
import argparse
import warnings

patient_master_list = [
    ["06348578",["7942GT00","7942GU00", "7942GZ00"]],
    ["05501184",["7913ZY00"]],
    ["05418761",["7972M300"]],
    ["06394294",["7942QQ00","7942QR00","00200200"]],
    ["05467817",["7971Q500"]],
    ["05323733",["7971UI00"]],
    ["05486196",["7941M700"]],
    ["05514820",["7941LY00"]],
    ["05513119",["00200100"]],
    ["02267738",["7941EC00"]],
    ["05447543",["7941D100","7941D200","7941D300","7941D400","7941D500","7941D600"]],
    ["02268547",["7940K700"]],
    ["05454991",["79410R00"]],
    ["05497695",["7941E901","7941EA00","7971C500","7971C700"]],
    ["05463487",["79519000"]],
    ["05235825",["79719R00"]],
    ["05352576",["7970IA00","7940HO00","7940HU00","22200Q01","22200S00","22200T00"]],
    ["05512494",["7971H000"]],
    ["05489744",["7941CO00","7951EX00"]],
    ["06381028",["7953A100","7953A400"]],
    ["00582992",["77952GP00"]],
    ["06338772",["7972OT00"]],
    ["05109836",["7952DG00","7952DP00","00108500","00108700"]],
    ["06452950",["7964JU00"]],
    ["00913367",["00105B00","00105C00","7962TA00"]]
    ]

def find_best_top1_file(directory):
    for filename in os.listdir(directory):
        if filename.startswith("best_top1"):
            return os.path.join(directory, filename)
    KeyError(f"No best_top1_acc file found in {directory}") 

def find_patient_index(patient_master_list, target_sub_id):
    for index, (patient_id, sub_ids) in enumerate(patient_master_list):
        if target_sub_id in sub_ids:
            return str(f"fold_{index + 1}")
    KeyError(f"No patient matching VideoID: {target_sub_id}") 

def unix_to_windows_path(path):
    return path.replace('\\', '/')

def main(config, work_dir, output_dir):
    video_dir = os.path.join(os.getcwd(), "annotations", "video_test_annotations")
    annotation_files = []
    patients = []
    for dirpath, dirnames, filenames in os.walk(video_dir):
        for filename in filenames:
            patients.append(filename.strip(".txt"))
            annotation_files.append(unix_to_windows_path(os.path.join(dirpath, filename)))
    
    for i, annotation in enumerate(annotation_files):
        test_ann = annotation
        fold_output_dir = os.path.join(output_dir, f'video_{patients[i]}')
        os.makedirs(fold_output_dir, exist_ok=True)

        # Create a temporary config file for this fold
        temp_config = os.path.join(fold_output_dir, f"i3d_config_{patients[i]}.py")

        checkpoint_to_test = unix_to_windows_path(find_best_top1_file(os.path.join(os.getcwd(), "LOPO_UNCROPPED_work_dirs", find_patient_index(patient_master_list, str(patients[i])))))
        print(f"\nTesting Video {patients[i]}")
        print("-"*30)
        print(f"Checkpoint: {checkpoint_to_test}\nOutput Dir: {fold_output_dir}")
        print("-"*30)
        
        with open(temp_config, 'w') as f:
            f.write(f"""
base_path = "C:/Users/u251245/CVEpilepsy/src/mmaction2/configs/_base_/"
data_path = "C:/Users/u251245/CVEpilepsy/video_clips_cropped/"
model_path = "{checkpoint_to_test}"
checkpoint_path = "{unix_to_windows_path(output_dir)}"

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
        raw_output_file = os.path.join(fold_output_dir, "raw_output.json")
        command = f"python tools/test.py {temp_config} {checkpoint_to_test} --out {raw_output_file} --eval save_diagnostics" 
        warnings.filterwarnings("ignore", category=UserWarning, module='mmcv')

        try:
            subprocess.run(command, check=True, shell=True)
            print(f"\nTesting succeeded for video {patients[i]}. Output saved to {fold_output_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Testing failed for fold {patients[i]} with error: {e}")
            continue

    print("\n\nTesting for all folds completed. Exiting...")

if __name__ == "__main__":
    cwd = os.getcwd()
    work_default = os.path.join(cwd, "LOPO_test_work_dir")
    output_default = os.path.join(cwd, "LOPO_test_output_dir")
    parser = argparse.ArgumentParser(description="Test MMAction2 models for 4-fold cross-validation.")
    parser.add_argument('--work-dir', default=work_default, help='Directory where testing outputs are saved')
    parser.add_argument('--output-dir', default=output_default, help='Directory where raw outputs are saved')
    config = ""
    args = parser.parse_args()
    main(config, args.work_dir, args.output_dir)