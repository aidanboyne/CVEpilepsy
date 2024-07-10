### Video annotation

For each seizure video, we mark the time period that a seizure was picked up on EEG and (optionally) when it was visibly apparent in a text file. It will likley be very difficult for the model to pick up on seizures without much or any movement.

For example, you could structure the annotation as a dictionary like this:

`('Video_path.mp4', 'video_ID', patient_ID, [start_EEG, end_EEG], [start_visual, end_visual])`

With a real patient it would look something like:

`('Videos/7942GT00.mp4', '7942GT00', 06348578, [306, 365], [312, 365])`

Once you've annotated all the videos and saved them into a text file, you can write a script to cut up the videos into 3-second segments and automatically assign annotations that are compatible with MMAction2 [[docs]](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html#action-recognition).

Here's an example I wrote up. Anthony did something similar but I had trouble understanding where things were being pulled from and saved, so I wrote a new one:

```
import cv2
import os

def clip_video(video_data, output_dir,
               output_eeg:str = "annotations_eeg.txt",
               output_vis:str = "annotations_vis.txt"):
    
    # Data unpacking
    video_path, video_id, patient_id, eeg_timestamps, visual_timestamps = video_data
    start_eeg, end_eeg = eeg_timestamps
    start_vis, end_vis = visual_timestamps

    cap = cv2.VideoCapture(video_path)

    # Get frames for a 3 second clip
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(fps * 3)
    os.makedirs(output_dir, exist_ok=True)

    # Open the output file and write clip to file

    current_frame = 0
    clip_count = 0
    while current_frame < total_frames:
        frames = []
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            current_frame += 1
        if len(frames) == clip_frames:
            clip_count += 1
            clip_start = (clip_count - 1) * clip_frames / fps
            clip_end = clip_start + 3 # MODIFY THIS FOR OVERLAPPING VIDEOS

            is_eeg_clip = start_eeg <= clip_start < end_eeg or start_eeg < clip_end <= end_eeg
            is_vis_clip = start_vis <= clip_start < end_vis or start_vis < clip_end <= end_vis

            # Save clip and write annotations to the annotation files
            clip_path = os.path.join(output_dir, f"{patient_id}_{video_id}_{clip_count}.mp4")
            out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
            for frame in frames:
                out.write(frame)
            out.release()

        with open(output_eeg, 'a') as f:
                f.write(f"{patient_id}_{video_id}_{clip_count}.mp4 {int(is_eeg_clip)}\n")
        with open(output_vis, 'a') as f:
               f.write(f"{patient_id}_{video_id}_{clip_count}.mp4 {int(is_vis_clip)}\n")

    cap.release()
    cv2.destroyAllWindows()
```

For a single video, you would run the clip with something like:

```
base_path = 'C:/Users/aidan/OneDrive - Baylor College of Medicine/Documents/BaylorMS1/Haneef/AnthonyCNN/Test_Datset_Gen/'
video_data = ('00100S00.mp4', '00100S00', 'Patient123', [18, 60], [20, 60])
clip_video(video_data=video_data, output_dir=base_path+'Clips')
```

Note that the script above cuts up the videos into non-overlapping clips. If you want more data for training, you can modify clip end time. Anthony used overlapping clips for training and testing, but not for validation.

### Train, Test, and Validation Datasets

Now that you have clipped all of the videos and have a master list of annotations, you have to split these into test/train/validation for each fold of cross validation. Lots of ways to do this: Anthony uses the random library which is pretty simple:

`train_videos = random.sample(train_val, k=21)`

Where train_val is just a list of all the video IDs and k is the number of IDs used for training. This might lead to pretty unbalanced data if certain videos are longer or have a longer proportion of video occupied by seizure activity so I would reccomend trying somthing a bit more tailored for this kind of stuff like StratifiedKFold from SciPy to take care of all of the clip assignment in one go [[docs]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

### Next steps

It will probably take a while to get all the annotations done and code up the above. After that, though, the only difficult part is setting up all of the config files for each model that will be trained in the k-fold cross validation. I think Anthony did this manually but it seems like that would take forever. I think we can tackle this when we get to it.