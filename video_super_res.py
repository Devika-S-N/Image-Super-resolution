import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Downscale/video-frame stitching pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--decode",
        action="store_true",
        help="Downscale HR video in Video/ and extract HR/LR frames"
    )
    group.add_argument(
        "--encode",
        action="store_true",
        help="Stitch frames from results/ into a video in Video/"
    )
    parser.add_argument(
        "-r", "--framerate",
        type=int,
        default=25,
        help="Frame rate for encoding when stitching frames"
    )
    return parser.parse_args()


def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def decode():
    video_dir = 'Video'
    if not os.path.isdir(video_dir):
        print(f"Error: '{video_dir}' folder not found.", file=sys.stderr)
        sys.exit(1)

    exts = ['.mp4', '.mkv', '.mov', '.hevc', '.avi']
    files = [f for f in os.listdir(video_dir)
             if os.path.splitext(f)[1].lower() in exts]
    if not files:
        print(f"No video file found in '{video_dir}'.", file=sys.stderr)
        sys.exit(1)

    hr_file = files[0]
    hr_path = os.path.join(video_dir, hr_file)
    base, ext = os.path.splitext(hr_file)

    lr_file = f"{base}_lr{ext}"
    lr_path = os.path.join(video_dir, lr_file)
    hr_frames = os.path.join(video_dir, 'HR_Frames')
    lr_frames = os.path.join(video_dir, 'LR_Frames')
    os.makedirs(hr_frames, exist_ok=True)
    os.makedirs(lr_frames, exist_ok=True)

    run_cmd([
        'ffmpeg', '-i', hr_path,
        '-vf', 'scale=iw/4:-2',
        '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
        '-an', lr_path
    ])

    run_cmd([
        'ffmpeg', '-i', hr_path,
        '-vsync', '0',
        os.path.join(hr_frames, 'frame_%06d.png')
    ])

    run_cmd([
        'ffmpeg', '-i', lr_path,
        '-vsync', '0',
        os.path.join(lr_frames, 'frame_%06d.png')
    ])

    print(f"Decode complete. LR video: {lr_path}")
    print(f"HR frames in: {hr_frames}, LR frames in: {lr_frames}")


def encode(framerate):
    results_dir = 'results_video'
    video_dir = 'Video'
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' folder not found.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(video_dir, exist_ok=True)

    out_name = os.path.basename(os.path.normpath(results_dir)) + '.mp4'
    out_path = os.path.join(video_dir, out_name)
    pattern = os.path.join(results_dir, 'sample_%01d.png')

    run_cmd([
        'ffmpeg', '-framerate', str(framerate),
        '-i', pattern,
        '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
        out_path
    ])

    print(f"Encode complete. Video saved to: {out_path}")


def main():
    args = parse_args()
    try:
        if args.decode:
            decode()
        elif args.encode:
            encode(args.framerate)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
