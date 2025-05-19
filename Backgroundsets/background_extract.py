import cv2
import os
import logging

root_path = ''
img_path = os.path.join(root_path, 'images')
mask_path = os.path.join(root_path, 'mask_batch_1_ucf')
save_path = os.path.join(root_path, 'backgrounds_train')
# Load images
logging.basicConfig(filename='background.log',level=logging.INFO)

for dir in sorted(os.listdir(img_path)):
    if os.path.isdir(os.path.join(img_path,dir)):
        for video_path in sorted(os.listdir(os.path.join(img_path,dir))):
            if os.path.isdir(os.path.join(img_path,dir,video_path)):
                for f in sorted(os.listdir(os.path.join(img_path,dir,video_path))):
                    if f.endswith('.jpg') or f.endswith('.png') :
                        img = os.path.join(img_path,dir,video_path,f)

                        original_image = cv2.imread(img)
                        
                        mask = img.replace('frames','mask')
                        mask = mask.replace('jpg','png')
                        if os.path.exists(mask):
                            mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                            if mask_image.dtype != 'uint8':
                                mask_image = mask_image.astype('uint8')
                            
                            if original_image.dtype != 'uint8':
                                original_image = original_image.astype('uint8')

                            inverted_mask = cv2.bitwise_not(mask_image)
                            background = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)
                            inpaint_radius = 3
                            background_inpaint = cv2.inpaint(background, mask_image, inpaint_radius, cv2.INPAINT_TELEA)
                            
                            save_dir = os.path.join(save_path,dir,video_path)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            cv2.imwrite(os.path.join(save_dir,f), background_inpaint)
                        else:
                            logging.info(f'lack mask {mask}')
 