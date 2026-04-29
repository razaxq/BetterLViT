import pandas as pd
import os
import shutil

df_val = pd.read_excel('datasets/Covid19/Val_ID.xlsx')
train_img_dir = 'datasets/Covid19/Train_Folder/img'
train_mask_dir = 'datasets/Covid19/Train_Folder/labelcol'
val_img_dir = 'datasets/Covid19/Val_Folder/img'
val_mask_dir = 'datasets/Covid19/Val_Folder/labelcol'

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

moved_count = 0
for mask_name in df_val['Image']:
    img_name = mask_name.replace('mask_', '')
    train_img_path = os.path.join(train_img_dir, img_name)
    train_mask_path = os.path.join(train_mask_dir, mask_name)
    
    if os.path.exists(train_img_path) and os.path.exists(train_mask_path):
        shutil.move(train_img_path, os.path.join(val_img_dir, img_name))
        shutil.move(train_mask_path, os.path.join(val_mask_dir, mask_name))
        moved_count += 1

print(f"Successfully moved {moved_count} samples to Val_Folder")
