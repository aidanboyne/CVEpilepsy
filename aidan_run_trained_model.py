import argparse
import os
import numpy as np
import cv2

"""
Model to JSON: python tools/test.py i3d_gaussian/i3d_config_00582992.py checkpoints/train_all/00582992/best_top1_acc_epoch_1.pth --out test1.json
Model to Figures: python tools/test.py i3d_gaussian/i3d_config_00582992.py checkpoints/train_all/00582992/best_top1_acc_epoch_1.pth --eval save_diagnostics

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run mmaction2 video classification model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model .pth file')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--config-path', type=str, required=True, help='Path to the model config file')
    args = parser.parse_args()
    return args

def generate_gradcam(model, video, device, target_layer_name='backbone/layer4'):
    # Register hooks to get gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    video = scatter(collate([video], samples_per_gpu=1), [device])[0]
    output = model(return_loss=False, **video)

    # Backward pass
    class_idx = output.argmax()
    model.zero_grad()
    output[0][class_idx].backward()

    grad = gradients[0].cpu().data.numpy()
    act = activations[0].cpu().data.numpy()

    # Generate Grad-CAM
    weights = np.mean(grad, axis=(2, 3))
    cam = np.zeros(act.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (video['imgs'][0].shape[2], video['imgs'][0].shape[1]))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def apply_colormap_on_image(org_im, activation, colormap_name):
    # Get colormap
    colormap = cv2.COLORMAP_JET
    if isinstance(colormap_name, int):
        colormap = colormap_name

    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(org_im)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    args = parse_args()

    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Build the model from a config file and a checkpoint file
    config = mmcv.Config.fromfile(args.config_path)
    model = init_recognizer(config, args.model_path, device=device)

    # Load video
    video_reader = mmcv.VideoReader(args.video_path)
    frames = [frame for frame in video_reader]

    # Inference and Grad-CAM
    data = dict(imgs=frames, label=-1)
    cam = generate_gradcam(model, data, device)

    # Save the Grad-CAM video
    gradcam_frames = []
    for frame in frames:
        frame = frame[:, :, ::-1]
        gradcam_frame = apply_colormap_on_image(frame, cam, cv2.COLORMAP_JET)
        gradcam_frames.append(gradcam_frame)

    output_path = os.path.splitext(args.video_path)[0] + '_gradcam.mp4'
    mmcv.frames2video(gradcam_frames, output_path, fps=video_reader.fps)

    # Inference
    results = inference_recognizer(model, args.video_path)

    print('Model Predictions:', results)
    print('Grad-CAM video saved to:', output_path)

if __name__ == '__main__':
    main()
