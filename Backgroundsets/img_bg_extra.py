import cv2
import numpy as np
import os
import glob

input_dir = ''
output_dir = ''

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = glob.glob(os.path.join(input_dir, '*.*'))
supported_formats = ('.jpg', '.jpeg', '.png', '.JPEG', '.tiff')

image_files = [f for f in image_files if f.lower().endswith(supported_formats)]

for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f'can not read images: {image_path}')
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width = img.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

    try:
        cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f'error: {image_path}\n: {e}')
        continue

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img_result = img_rgb * mask2[:, :, np.newaxis]
    img_output = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_name)

    cv2.imwrite(output_path, img_output)
    print(f'save images: {output_path}')
