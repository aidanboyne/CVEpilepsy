import os
import subprocess
import json
import argparse

# To run script: python aidan_train_all_models.py patient1 patient2 patient3 ... --output-dir path/to/output
# On my machine: python aidan_train_all_models.py 00582992 00913367 02267738 02268547 05109836 05235825 05323733 05352576 05418761 05447543 05454991 05463487 05467817 05486196 05489744 05497695 05501184 05512494 05513119 05514820 06338772 06348578 06381028 06394294 06452950 --output-dir C:/Users/u251245/CVEpilepsy/checkpoints/train_all/

def main(patient_ids, output_dir):
    for patient_id in patient_ids:
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        config_file = f"i3d_gaussian/i3d_config_{patient_id}.py"
        command = f"python tools/train.py {config_file} --validate --gpus 1 --work-dir {patient_output_dir}"
        
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"\nTraining succeeded for patient {patient_id}. Output saved to {patient_output_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Training failed for patient ID {patient_id} with error: {e}")
            continue

    print("\n\nTraining for all patients completed. Exiting...")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MMAction2 models for a list of patient IDs.")
    parser.add_argument('patient_ids', nargs='+', help='List of patient IDs to train models for')
    parser.add_argument('--output-dir', default='work_dirs', help='Directory where training outputs are saved')
    
    args = parser.parse_args()
    main(args.patient_ids, args.output_dir)
