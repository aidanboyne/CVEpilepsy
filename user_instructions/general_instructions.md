### Set up the environnment

MMAction2 Framework
20-min tutorial: if you don't have specs to run this you might have to use one of Dr. Haneef's machines

Write program to automatically clip the videos and create annotation file for model training.

Download the videos and annotation key from the Teams Chat Folder (ImageRasna_CNN)

Modify the example code I wrote for another collaborator (video_dataset_outline.md) to take the input from the annotation key(video_notes.txt) 
and use it to cut the videos into clips and create the annotation file

Now try to create a config.py file
Docs here

Example config.py from Anthony is in the Teams folder
Base model that you are retraining is the .pth file in the Teams folder

### Video annotation

For each seizure video, we mark the time period that a seizure was picked up on EEG and (optionally) when it was visibly apparent in a text 
file. It will likley be very difficult for the model to pick up on seizures without much or any movement.

For example, you could structure the annotation as a dictionary like this:

`('Video_path.mp4', 'video_ID', patient_ID,  [start_visual, end_visual])`

With a real patient it would look something like:

`('Videos/7942GT00.mp4', '7942GT00', 06348578,  [312, 365])`

Once you've annotated all the videos and saved them into a text file, you can write a script to cut up the videos into 3-second segments and 
automatically assign annotations that are compatible with MMAction2 [[docs]](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html#action-recognition).


### Train, Test, and Validation Datasets

Now that you have clipped all of the videos and have a master list of annotations, you have to split these into test/train/validation for 
each fold of cross validation. Lots of ways to do this: Anthony uses the random library which is pretty simple:

`train_videos = random.sample(train_val, k=24)`

Where train_val is just a list of all the video IDs and k is the number of IDs used for training. This might lead to pretty unbalanced data 
if certain videos are longer or have a longer proportion of video occupied by seizure activity so I would reccomend trying somthing a bit 
more tailored for this kind of stuff like StratifiedKFold from SciPy to take care of all of the clip assignment in one go 
[[docs]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).


### Next steps

It will probably take a while to get all the annotations done and code up the above. After that, though, the only difficult part is setting 
up all of the config files for each model that will be trained in the k-fold cross validation. I think Anthony did this manually but it 
seems like that would take forever. I think we can tackle this when we get to it.

### setting up all of the config files for each model

1. _base_: find the corresponding i3d_r50.py, sgd_100e.py, and default_runtime.py files. 

2. They should be in a similar location but the first few parts (e.g. Uses/Aidan/mmaciton2...) will be different for your computer. 
Change these to your locations

3. Dataset settings: Change filepaths for the dataset settings (data_root, data_root_val,.... ann_file_test). 

4. I would recommend putting all of the video clips in a single folder. Then the folder for data_root, data_root_val, and data_root_test will all be the same folder (whatever folder you pick). 

5. For the ann_file_train/val/test: pick the annotation files for a single patient that you left out and provide those paths (doesn't matter \
which one right now, you can just use train_00582992.txt, val_00582992.txt, and test_00582992.txt) 

6. load_from: This is the path to the model we are going to train. Again, should be in similar location to what is already in the file but the beginning of the path (Users/Aidan/...) will be different for your computer. Make sure the path points to the file 
i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth

7. Finally, change the path in work_dirto anything you want. 
Once this is done, we can try and train the model. 

8. You will have to use the terminal to activate the environment you set up (Anthony's environment), cd to the main project directory, 
then run the command `python tools/train.py configs/i3d_config_base.py --validate --gpus 1`