import os
from pathlib import Path
from dwpose.preprocess import get_image_pose
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def process_image(input_image_path, output_image_path):
    """
    处理单张图片，进行姿态估计，并保存结果图。

    Args:
        input_image_path (Path): 输入图片路径。
        output_image_path (Path): 输出图片路径。

    Returns:
        bool: 如果处理成功，返回True；否则，返回False。
    """
    try:
        # 使用 Pillow 读取图像
        with Image.open(input_image_path) as img:
            # 如果图像有Alpha通道，去除Alpha通道
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            ref_image_rgb = np.array(img)

        # 调用 get_image_pose 方法
        pose_image = get_image_pose(ref_image_rgb)

        # 检查返回的图像形状并调整维度
        if pose_image.shape[0] == 3:
            # 通道在前，需要转置
            pose_image = np.transpose(pose_image, (1, 2, 0))
        elif pose_image.shape[2] != 3:
            # 如果通道数不为3，无法处理
            return False

        # 确保输出目录存在
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 Matplotlib 保存图像
        plt.imsave(output_image_path, pose_image)
        return True

    except Exception as e:
        # 如果处理过程中出现任何异常，返回False
        return False

def main():
    # 基础目录（假设脚本与TikTok_event同级）
    base_dir = Path('/root/autodl-tmp/TikTok_event')

    # 定义要处理的集和对应的输出集
    sets = {
        'train_set': 'dwpose_train_set',
        'val_set': 'dwpose_val_set',
        'test_set': 'dwpose_test_set'
    }

    # 初始化失败列表
    failed_images = []

    # 遍历每个集
    for input_set, output_set in sets.items():
        input_set_path = base_dir / input_set
        output_set_path = base_dir / output_set

        # 检查输入集目录是否存在
        if not input_set_path.exists():
            print(f"输入目录 {input_set_path} 不存在，跳过。")
            continue

        print(f"正在处理集：{input_set}")

        # 获取所有视频子目录
        video_dirs = [d for d in input_set_path.iterdir() if d.is_dir()]

        # 统计总图片数
        total_images = sum(len([f for f in video_dir.glob('*.png') if not f.name.startswith('.')]) for video_dir in video_dirs)

        # 使用tqdm创建进度条
        with tqdm(total=total_images, desc=f"Processing {input_set}") as pbar:
            # 遍历每个视频子目录
            for video_dir in video_dirs:
                # 相对路径以保持目录结构
                relative_video_dir = video_dir.relative_to(input_set_path)
                output_video_dir = output_set_path / relative_video_dir

                # 获取所有.png图片，排除以.开头的隐藏文件
                image_files = [f for f in video_dir.glob('*.png') if not f.name.startswith('.')]

                for image_file in image_files:
                    # 定义输出图片路径
                    output_image_file = output_video_dir / image_file.name

                    # 处理图片
                    success = process_image(image_file, output_image_file)

                    if not success:
                        # 记录失败的图片路径（相对于base_dir）
                        failed_images.append(str(image_file.relative_to(base_dir)))

                    # 更新进度条
                    pbar.update(1)

    # 将失败的图片路径写入failed_images.txt
    if failed_images:
        failed_file = base_dir / 'failed_images.txt'
        with failed_file.open('w') as f:
            for path in failed_images:
                f.write(f"{path}\n")
        print(f"\n处理完成，但有 {len(failed_images)} 张图片处理失败。详情见 {failed_file}")
    else:
        print("\n所有图片均已成功处理。")

if __name__ == "__main__":
    main()