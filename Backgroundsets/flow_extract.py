import cv2
import numpy as np
import os

def extract_optical_flow(frames_path, flow_output_path):
    if not os.path.exists(flow_output_path):
        os.makedirs(flow_output_path)

    frames = sorted([os.path.join(frames_path, frame) for frame in os.listdir(frames_path) if frame.endswith('.jpg') or frame.endswith('.png')])
    
    if not frames:
        print(f"No image files found in {frames_path}. Skipping...")
        return
    
    prev_frame = cv2.imread(frames[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    for idx, frame_path in enumerate(frames[1:], start=1):
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # 将光流转换为HSV表示
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 保存光流图像
        cv2.imwrite(os.path.join(flow_output_path, f"flow_{idx:04d}.png"), bgr)
        
        prev_gray = gray
    
    print(f"Optical flow extracted and saved to {flow_output_path}")

def process_dataset(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        if files:
            frames_path = root
            relative_path = os.path.relpath(frames_path, dataset_path)
            flow_output_path = os.path.join('/data/ucf101/flows/', relative_path)
            extract_optical_flow(frames_path, flow_output_path)

dataset_path = '/data/ucf101/frames/'
process_dataset(dataset_path)
