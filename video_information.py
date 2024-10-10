import cv2
import os
import re

# MAKE SURE PATH IS CORRECT IN LINE 93!
# Returns information for each patient (e.g. video length, percent seizure...) based on an annotation file.
# Requires annotations file is already created

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return frame_count, fps

def process_patient_videos(file_path):
    patient_data = {}
    current_patient = None
    total_frames = 0
    total_seizure = 0
    total_non_seizure = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Patient'):
                current_patient = line.strip()
                patient_data[current_patient] = {
                    'total_frames': 0,
                    'seizure_frames': 0,
                    'non_seizure_frames': 0,
                    'videos': []
                }
            elif line.startswith('('):
                match = re.match(r"\('(.+\.mp4)', (\d+), (\d+), '(.+)', sec\)", line.strip())
                if match:
                    video_path, start_sec, end_sec, _ = match.groups()
                    start_sec = int(start_sec)
                    end_sec = int(end_sec)
                    
                    frame_count, fps = get_video_info(video_path)
                    if frame_count and fps:
                        seizure_frames = (end_sec - start_sec) * fps
                        non_seizure_frames = frame_count - seizure_frames
                        
                        total_frames += frame_count
                        total_non_seizure += non_seizure_frames
                        total_seizure += seizure_frames

                        patient_data[current_patient]['total_frames'] += frame_count
                        patient_data[current_patient]['seizure_frames'] += seizure_frames
                        patient_data[current_patient]['non_seizure_frames'] += non_seizure_frames
                        
                        patient_data[current_patient]['videos'].append({
                            'video_path': video_path,
                            'total_frames': frame_count,
                            'fps':fps,
                            'seizure_frames': seizure_frames,
                            'non_seizure_frames': non_seizure_frames
                        })
    metadata = {
        'total_frames': total_frames,
        'seizure_frames': total_seizure,
        'nonseizure_frames': total_non_seizure
    }

    return [patient_data, metadata]

def write_patient_info(output_file, patient_data, metadata):
    with open(output_file, 'w') as out_file:
        for patient, data in patient_data.items():
            out_file.write(f"{patient}\n")
            out_file.write(f"Total Frames: {data['total_frames']}\n")
            out_file.write(f"Seizure Frames: {data['seizure_frames']}\n")
            out_file.write(f"Non-Seizure Frames: {data['non_seizure_frames']}\n")
            out_file.write("Videos:\n")
            for video in data['videos']:
                out_file.write(f"  Video Path: {video['video_path']}\n")
                out_file.write(f"  Total Frames: {video['total_frames']}\n")
                out_file.write(f"FPS: {video['fps']}\n")
                out_file.write(f"  Seizure Frames: {video['seizure_frames']}\n")
                out_file.write(f"  Non-Seizure Frames: {video['non_seizure_frames']}\n")
            out_file.write("\n\n")
        out_file.write(f"""Total Frames: {metadata['total_frames']}\nSeizure Frames:{metadata['seizure_frames']:.0f}\t{((metadata['seizure_frames']/metadata['total_frames'])*100):.2f}%\nNon-seizure Frames:{metadata['nonseizure_frames']:.0f}\t{((metadata['nonseizure_frames']/metadata['total_frames'])*100):.2f}%""")
    
    print(f"Patient information has been written to {output_file}")

if __name__ == "__main__":
    input_text_file = os.path.join(os.getcwd(), "annotations", "usuable_data.txt")
    output_text_file = "./video_info.txt"
    
    patient_data, metadata = process_patient_videos(input_text_file)
    write_patient_info(output_text_file, patient_data, metadata)
