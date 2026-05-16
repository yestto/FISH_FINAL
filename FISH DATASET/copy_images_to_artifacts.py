import cv2
import glob
import os
import shutil

src_root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
dst_dir = r'C:\Users\shain\.gemini\antigravity\brain\87351874-ce08-4f2d-8799-83af16d672a0\scratch\fish_images'
os.makedirs(dst_dir, exist_ok=True)

fish_dirs = ['fish01','fish2','fish3','fish4','fish5','fish6','fish7',
             'fish8','fish9','fish10','fish11','fish12','fish13','fish14','fish15']

for fdir in fish_dirs:
    img_folder = os.path.join(src_root, fdir, 'single image')
    if os.path.isdir(img_folder):
        imgs = glob.glob(os.path.join(img_folder, '*.jpg'))
        if imgs:
            img_path = imgs[0]
            dst_path = os.path.join(dst_dir, f"{fdir}.jpg")
            
            # Read and resize to 800px max dimension for easier viewing
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                scale = 800.0 / max(h, w)
                resized = cv2.resize(img, (int(w*scale), int(h*scale)))
                cv2.imwrite(dst_path, resized)

print("Images copied to artifacts scratch folder.")
