import os
import yaml
from ultralytics import YOLO
import torch
import multiprocessing 


DATASET_ROOT_DIR = 'datasets/meter_readings'
YAML_FILE_PATH = os.path.join(DATASET_ROOT_DIR, 'meter_data.yaml')
MODEL_VARIANT = 'yolov8s.pt'
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8
PROJECT_NAME = 'meter_detection_runs'
EXPERIMENT_NAME = 'exp_reading_area_detector' 


if __name__ == '__main__': 
    multiprocessing.freeze_support()

    train_images_path = os.path.abspath(os.path.join(DATASET_ROOT_DIR, 'images', 'train'))
    val_images_path = os.path.abspath(os.path.join(DATASET_ROOT_DIR, 'images', 'val'))

    if not os.path.isdir(train_images_path):
        print(f"错误: 训练图片目录未找到: {train_images_path}")
        exit()
    if not os.path.isdir(val_images_path):
        print(f"错误: 验证图片目录未找到: {val_images_path}")
        exit()

    data_yaml_content = {
        'train': train_images_path,
        'val': val_images_path,
        'nc': 1,
        'names': ['reading_area']
    }

    try:
        with open(YAML_FILE_PATH, 'w') as f:
            yaml.dump(data_yaml_content, f, sort_keys=False)
        print(f"成功创建/更新 data.yaml 文件: {YAML_FILE_PATH}")
        print(yaml.dump(data_yaml_content, sort_keys=False))
    except Exception as e:
        print(f"错误: 无法创建 data.yaml 文件: {e}")
        exit()


    try:
        model = YOLO(MODEL_VARIANT)
        print(f"成功加载模型: {MODEL_VARIANT}")
    except Exception as e:
        print(f"错误: 加载模型 {MODEL_VARIANT} 失败: {e}")
        exit()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"将使用设备: {device.upper()}")
    if device == 'cuda':
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    print("\n开始模型训练...")
    try:
        results = model.train(
            data=YAML_FILE_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME, 
            device=device,
            patience=20,
            workers=1,
            exist_ok=True,

        )
        print("模型训练完成!")
        final_save_dir = results.save_dir if hasattr(results, 'save_dir') else os.path.join(PROJECT_NAME, EXPERIMENT_NAME)
        print(f"训练结果保存在: {final_save_dir}")
        print(f"最佳模型权重保存在: {os.path.join(final_save_dir, 'weights', 'best.pt')}")


    except Exception as e:
        print(f"模型训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n开始模型评估...")
    try:
        best_model_path = os.path.join(final_save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            trained_model = YOLO(best_model_path)
            metrics = trained_model.val(
                data=YAML_FILE_PATH,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                device=device,
                split='val'
            )
            print("模型评估完成。")
        else:
            print(f"未找到训练好的最佳模型: {best_model_path}，跳过评估。")
    except Exception as e:
        print(f"模型评估过程中发生错误: {e}")