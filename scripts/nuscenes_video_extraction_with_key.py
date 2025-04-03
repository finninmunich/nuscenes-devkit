import os
import cv2
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm import tqdm
def render_sample_data(nusc, sample_data_token, output_image_path, with_anns=True, box_vis_level=BoxVisibility.ANY):
    # 获取传感器类型
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']

    if sensor_modality == 'camera':
        # 加载box和图像
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
        data = Image.open(data_path)

        # 创建图像和轴
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        ax.imshow(data)

        # 显示box
        if with_anns:
            for box in boxes:
                c = np.array(nusc.explorer.get_color(box.name)) / 255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
                # 在box上添加token
                #corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                #ax.text(corners[0, 0], corners[1, 0], box.token, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)
        ax.axis('off')
        ax.set_aspect('equal')

        # 将图像保存到指定路径
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)
    else:
        raise ValueError("错误：未知的传感器类型！")

def render_scene_to_images(nusc, scene_token, output_dir):
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    # 获取相机通道
    cam_channel = 'CAM_FRONT'

    # 获取相机通道的所有样本数据token
    cam_tokens = []
    while sample:
        cam_tokens.append(sample['data'][cam_channel])
        if sample['next'] == '':
            break
        sample = nusc.get('sample', sample['next'])

    # 渲染每一帧并保存为图像
    for i, cam_token in enumerate(cam_tokens):
        output_image_path = os.path.join(output_dir, f"{i:06d}.jpg")
        render_sample_data(nusc, cam_token, output_image_path)

def images_to_video(image_dir, output_video_path, fps=2):
    # 获取所有图像文件
    images = [img for img in os.listdir(image_dir) if img.endswith(".jpg") and img != '000000.jpg']
    images.sort()
    # 读取第一张图像以获取视频尺寸
    first_image_path = os.path.join(image_dir, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 写入每一帧图像
    for image in images:
        image_path = os.path.join(image_dir, image)
        img = cv2.imread(image_path)
        video_writer.write(img)

    video_writer.release()

def process_scene(nusc, scene, output_dir):
    scene_token = scene['token']
    scene_output_dir = os.path.join(output_dir, 'image', scene['name'])
    os.makedirs(scene_output_dir, exist_ok=True)
    print(f"渲染场景{scene['name']}的图像到{scene_output_dir}")
    render_scene_to_images(nusc, scene_token, scene_output_dir)
    output_video_path = os.path.join(output_dir, 'videos', f"{scene['name']}.mp4")
    print(f"将图像合成为视频{output_video_path}")
    images_to_video(scene_output_dir, output_video_path)
    print(f"场景{scene['name']}渲染完成！")

def main(version, dataroot, output_dir,key='truck'):
    # 初始化NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # 如果输出目录不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    target_scene = []
    for scene in nusc.scene:
        scene_description = scene['description']
        if key == 'truck':
            if 'truck' in scene_description or 'Truck' in scene_description:
                target_scene.append(scene)
        elif key == 'overtake':
            if 'overtake' in scene_description or 'Overtake' in scene_description:
                target_scene.append(scene)
        elif key =='trailer':
            if 'trailer' in scene_description or 'Trailer' in scene_description:
                target_scene.append(scene)
        elif key =='construction':
            if 'construction vehicle' in scene_description or 'Construction vehicle' in scene_description:
                target_scene.append(scene)
        else:
            raise ValueError("错误：未知的关键字！")

    print(f"共有{len(target_scene)}个场景描述{key}。")
    #     if 'overtake' in scene_description or 'Overtake' in scene_description:
    #         target_scene.append(scene)
    # print(f"共有{len(target_scene)}个场景描述超车。")

    # 顺序处理每个场景
    for scene in tqdm(target_scene):
        process_scene(nusc, scene, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将NuScenes场景渲染为视频。")
    parser.add_argument('-v', '--version', type=str, default='v1.0-trainval', help='NuScenes数据集版本')
    parser.add_argument('-root', '--dataroot', type=str, default='/iag_ad_01/ad/finn/finn_data/nuscenes', help='NuScenes数据集路径')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='保存输出视频的目录')
    parser.add_argument('-k', '--key', type=str, default=None, help='筛选场景的关键字')

    args = parser.parse_args()
    main(args.version, args.dataroot, args.output_dir,args.key)