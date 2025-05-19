import cv2
import numpy as np
import os
import glob

def extract_and_inpaint_background(original_dir, foreground_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_files = glob.glob(os.path.join(original_dir, '*.*'))
    supported_formats = ('.jpg', '.jpeg', '.png', '.JEPG', '.tiff')
    original_files = [f for f in original_files if f.lower().endswith(supported_formats)]
    
    for original_path in original_files:
        base_name = os.path.basename(original_path)
        foreground_path = os.path.join(foreground_dir, base_name)
        
        if not os.path.exists(foreground_path):
            print(f'lack foreground_path: {foreground_path}')
            continue
        
        original_img = cv2.imread(original_path)
        foreground_img = cv2.imread(foreground_path)
        
        if original_img is None or foreground_img is None:
            print(f'can not read image: {base_name}')
            continue
        
        if original_img.shape != foreground_img.shape:
            print(f'shape error: {base_name}')
            continue
        
        foreground_mask = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_BINARY)
        mask = mask.astype('uint8')
        
        background = original_img.copy()
        inpaint_radius = 3
        filled_background = cv2.inpaint(background, mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        output_path = os.path.join(output_dir, base_name)
        cv2.imwrite(output_path, filled_background)
        print(f'save backgrounds: {output_path}')

original_dir = ''
foreground_dir = ''
output_dir = ''

extract_and_inpaint_background(original_dir, foreground_dir, output_dir)
