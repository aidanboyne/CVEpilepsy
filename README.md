# CV Epilepsy
#### Code used to produce results in **Three-Dimensional Convolutional Neural Network Based Detection of Epilectic Seizures from Video Data**
_Aidan Boyne, Hsiang J. Yeh, Anthony K. Allam, Brandon M. Brown, Mohammad Tabaeizadeh, John M. Stern, R. James Cotton, Zulfi Haneef_

Preprint: [medRxiv](https://www.medrxiv.org/content/10.1101/2024.10.11.24315247v1)
Model weights: [Storage Bucket](https://bcmedu-my.sharepoint.com/:f:/g/personal/u251245_bcm_edu/EugwlExA3IdFnzNzwUcAPXsB9Lf2KUR4Q53ruaWDXIcj_A?e=2T47nW)

For installation see _`Install.md`_

## How to use this repository

Use of this repo requires a set of target videos and corresponding annotations for training. Weights for models trained at the BCM and UCLA are also provided in the OneDrive bucket above.

Clone the repository and install depenencies in `requirements.txt`. Then, run code in the files listed below in order.

### annotate.ipynb

Run this first to create a readable file for the video clipping and clip annotations based on a main file with the manual annotations structured like:

```python
Patient # (number):
('Video/path.mp4', seizure_start_seconds, seizure_end_seconds, 'ID')
```
For example:
```python
Patient 1 (06348578):
('Videos/7942GT00.mp4', 306, 365, '7942GT00', sec) # significant movement 
('Videos/7942GU00.mp4', 300, 362, '7942GU00', sec) # significant movement
```

### clips.ipynb

Takes full videos and clips them into 3 second videos for training, testing, and evaluation. Also creates a master annoatation file which labels each clip as 1 (seizure) or 0 (non-seizure) based on the file created by `annotate.ipynb`.

This file is a bit buggy, so you have to rename the resulting clips (I reccomend using a mass find and replace in the file explorer using something like notepad++)

### video_information.py

Gives information about each video (length, percent seizure) if a corresponding annotation file is available and the video has been clipped.

### Kfold_CV.ipynb

If all annotations have been saved to a master file, makes subfolders of annotations for train, validation, test in model cross validation.

### train_KfoldCV.py

Based on cross-validation directores created by `Kfold_CV.ipynb`, automatically creates the necessary configuration files for each cross validation fold, trains the model, and evaluates the fold performance.

Models and raw log files are saved to a work directory. Performance evaluation is saved to the path specified in `src/mmaction2/mmaction/core/evaluation/accuracy.py`

In detail, the script:
1. Sets up the cross-validation directory structure.
2. Iterates through 25 folds (1 to 25).
3. For each fold:

    - Creates a fold-specific output directory.
    - Generates paths for train, validation, and test annotation files.
    - Creates a temporary configuration file for the current fold.
    - Writes a large configuration string to this file, which includes:
        - Paths to various directories and files.
        - Model parameters and settings.
        - Dataset configurations.
        - Training pipeline settings.
    - Executes a training command using subprocess, which runs a Python script (tools/train.py) with the generated config file.
    - Handles success or failure of the training process for each fold.


#### Command-line Interface:

The script uses argparse to handle command-line arguments.
It accepts an optional `--output-dir` argument to specify where training outputs should be saved.

Configuration File Content:
The generated configuration file for each fold includes:
- Base configurations for the model, schedule, and runtime.
- Model settings (using an I3D backbone with non-local blocks).
- Dataset settings (using a VideoDataset type).
- Data preprocessing pipelines for training, validation, and testing.
- Optimization settings.
- Evaluation metrics.
- Paths for loading pre-trained weights and saving checkpoints.

Training Execution: The script uses `subprocess.run()` to execute the training command for each fold.
It captures and prints any errors that occur during training.


### test.py

The `main()` function is the core of the script. It performs the following tasks:
1. Sets up the directory structure for test annotations.
2. Iterates through all annotation files in the specified directory.
3. For each annotation file (representing a patient's data):

- Creates a patient-specific output directory.
- Generates a temporary configuration file for the current patient.
- Writes a large configuration string to this file, which includes:
    - Paths to various directories and files.
    - Model parameters and settings.
    - Dataset configurations.
    - Testing pipeline settings.
- Executes a testing command using subprocess, which runs a Python script (tools/test.py) with the generated config file and specified checkpoint.
- Handles success or failure of the testing process for each patient.


Configuration File Content:
The generated configuration file for each patient includes:

- Base configurations for the model, schedule, and runtime.
- Model settings (using an I3D backbone with non-local blocks).
- Dataset settings (using a VideoDataset type).
- Data preprocessing pipeline for testing.
- Evaluation metrics.
- Paths for loading pre-trained weights and saving outputs.

Command-line Interface:

The script uses argparse to handle command-line arguments.
It accepts the following arguments:

- `checkpoint`: Path to the checkpoint file to be used for testing (required).
- `--work-dir`: Directory where testing outputs are saved (optional, has a default value).
- `--output-dir`: Directory where raw outputs are saved (optional, has a default value).

### test_LOPO.py

Specialized version of `test.py` to run testing by selecting the best model from training for each patient for proper cross-validation testing in leave-one-patient-out cross-validation schemes.

### figures_stats.ipynb

Notebook containing plotting and statistical analysis functions based primarily on the files produced during training or testing by the custom `save_diagnostics` function in the mmaction evaluation files.

### find_optimal_models.ipynb

After training, digs through all of the diagnostic files to find the best performing model (for use in testing) and outputs some brief stats concerning the training process.



