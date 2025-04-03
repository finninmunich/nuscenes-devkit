import os
import argparse
import subprocess


def convert_video(input_path, output_path):
    """
    使用 FFmpeg 将视频转换为 H.264 (libx264) 编码，并设置像素格式为 yuv420p。
    """
    command = [
        "ffmpeg", "-i", input_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"转换完成: {output_path}")


def batch_convert_videos(input_dir):
    """
    遍历目录，转换所有 .mp4 文件。
    """
    if not os.path.isdir(input_dir):
        print("错误: 目录不存在!")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(input_dir, f"converted_{filename}")
            convert_video(input_path, output_path)
            print(f"转换完成: {filename} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="转换目录中的所有 MP4 文件，使其兼容 QuickTime Player。")
    parser.add_argument("--input_dir", type=str, help="包含 MP4 文件的目录")
    args = parser.parse_args()

    batch_convert_videos(args.input_dir)


if __name__ == "__main__":
    main()
