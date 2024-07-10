### Step 1: Install MMAction2

First, ensure you have MMAction2 installed. You can install it using the following commands:

```bash
conda create -n mmaction2 python=3.8 -y
conda activate mmaction2
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
```

### Step 2: Prepare Your Data and Configuration

Ensure your dataset is prepared according to MMAction2's format and that you have a configuration file ready for your model.

### Step 3: Train Your Model

Train your model using MMAction2. For example:

```bash
python tools/train.py configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py
```

Replace the configuration file path with your specific configuration.

### Step 4: Evaluate Your Model and Obtain Predictions

Once your model is trained, you need to evaluate it to get the predicted probabilities and true labels. Use the following command to perform evaluation:

```bash
python tools/test.py configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py work_dirs/tsn_r50/latest.pth --eval top_k_accuracy mean_class_accuracy
```

### Step 5: Extract Predictions and Ground Truth

The evaluation output will contain the predictions and the ground truth labels. You can modify the evaluation script to save these values to a file. In `tools/test.py`, add code to save the predictions and ground truth after the evaluation is completed:

```python
# Add this import
import numpy as np
import os

# Modify the evaluation script
# After predictions and ground truth are obtained
np.save('predictions.npy', results)
np.save('labels.npy', gt_labels)
```

### Step 6: Calculate AUC-ROC

After saving the predictions and ground truth labels, you can calculate the AUC-ROC using any Python script. Here’s an example using scikit-learn:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the predictions and labels
predictions = np.load('predictions.npy')
labels = np.load('labels.npy')

# Assuming binary classification and predictions are probabilities
auc_roc = roc_auc_score(labels, predictions)
print(f'AUC-ROC: {auc_roc}')
```

If you have multi-class classification, you may need to calculate the AUC-ROC for each class separately or use an appropriate multi-class AUC-ROC function.

### Example for Multi-class AUC-ROC

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Binarize the labels for multi-class
labels_binarized = label_binarize(labels, classes=np.arange(num_classes))

# Calculate AUC-ROC for each class
auc_roc = roc_auc_score(labels_binarized, predictions, multi_class='ovo')
print(f'Multi-class AUC-ROC: {auc_roc}')
```

Replace `num_classes` with the actual number of classes in your dataset.

By following these steps, you can calculate the AUC-ROC for your model trained using the MMAction2 framework.

### Tensorboard

Run the following for install in your MMaction2 environment:
- `conda install pip`
- `pip install tensorboard`

Then when model is training, run the following command in a seperate terminal with the environment up:
- `tensorboard --logdir=C:/Users/u251245/CVEpilepsy/checkpoints/TensorBoard`
