import pandas as pd
from PIL import Image
import os
import shutil


csv_file_path = 'annotations.csv'
raw_images_source_dir = 'raw_images' 


output_dir_base = 'datasets/meter_readings'
images_dir_base = os.path.join(output_dir_base, 'images')
labels_dir_base = os.path.join(output_dir_base, 'labels')


if not os.path.exists(csv_file_path):
    print(f"错误: CSV文件 '{csv_file_path}' 未找到。")
    exit()
if not os.path.isdir(raw_images_source_dir):
    print(f"错误: 原始图片目录 '{raw_images_source_dir}' 未找到。")
    exit()


if os.path.exists(output_dir_base):
    print(f"警告: 输出目录 '{output_dir_base}' 已存在，将删除并重建。")
    shutil.rmtree(output_dir_base)


os.makedirs(os.path.join(images_dir_base, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir_base, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_dir_base, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir_base, 'val'), exist_ok=True)


df = pd.read_csv(csv_file_path)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(df_shuffled) * 0.80)
train_df = df_shuffled[:split_index]
val_df = df_shuffled[split_index:]

datasets = {'train': train_df, 'val': val_df}
print(f"数据划分完成: {len(train_df)} 个训练样本, {len(val_df)} 个验证样本。")


total_processed = 0
for dataset_type, current_df in datasets.items():
    print(f"\n--- 正在处理 {dataset_type} 数据集 ---")
    for index, row in current_df.iterrows():
        try:
            filename_with_ext = row['filename']
            base_filename = os.path.splitext(filename_with_ext)[0]

            image_path_source = os.path.join(raw_images_source_dir, filename_with_ext)
            image_path_dest = os.path.join(images_dir_base, dataset_type, filename_with_ext)
            label_path_dest = os.path.join(labels_dir_base, dataset_type, f"{base_filename}.txt")

            if not os.path.exists(image_path_source):
                print(f"  [警告] 源图片未找到，跳过: {image_path_source}")
                continue

            with Image.open(image_path_source) as img:
                img_width, img_height = img.size
            
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            if xmin >= xmax or ymin >= ymax:
                print(f"  [警告] 坐标无效 (xmin>=xmax or ymin>=ymax)，跳过: {filename_with_ext}")
                continue

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            class_index = 0

            with open(label_path_dest, 'w', encoding='utf-8') as f:
                f.write(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

            shutil.copy(image_path_source, image_path_dest)
            
            total_processed += 1
            if total_processed % 100 == 0: 
                 print(f"  已处理 {total_processed} 个文件...")

        except Exception as e:
            print(f"  [错误] 处理 {row.get('filename', '未知文件')} 时发生严重错误: {e}")


print(f"总共成功处理了 {total_processed} 个图像和标签。")
print(f"请检查以下目录中的文件是否已正确生成:")
print(f"- 训练图片: {os.path.join(images_dir_base, 'train')}")
print(f"- 验证图片: {os.path.join(images_dir_base, 'val')}")
print(f"- 训练标签: {os.path.join(labels_dir_base, 'train')}")
print(f"- 验证标签: {os.path.join(labels_dir_base, 'val')}")