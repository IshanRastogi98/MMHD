import os
import time
import json

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg


SOURCE_DIR = "source_videos"
OUTPUT_DIR = "sample_clips"

THRESHOLD = 27.0


def segment_video(video_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]

    output_folder = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(output_folder, exist_ok=True)

    report_path = os.path.join(
        output_folder,
        "segmentation_report.json"
    )

    if os.path.exists(report_path):
        print(f"Already segmented: {video_name}")
        return

    print(f"\nProcessing: {filename}")

    start_time = time.perf_counter()

    video = open_video(video_path, backend="pyav")

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=THRESHOLD)
    )

    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    detection_time = time.perf_counter() - start_time

    print(f"Detected scenes: {len(scenes)}")
    print(f"Detection time: {detection_time:.2f} seconds")

    if scenes:
        split_video_ffmpeg(
            video_path,
            scenes,
            output_dir=output_folder,
            arg_override="-map 0 -c copy",
            show_progress=True
        )

    durations = []

    for start_timecode, end_timecode in scenes:
        duration = (
            end_timecode.get_seconds()
            - start_timecode.get_seconds()
        )
        durations.append(duration)

    report = {
        "source_video": filename,
        "scene_count": len(scenes),
        "scene_durations_seconds": durations,
        "average_scene_duration_seconds": (
            sum(durations) / len(durations)
            if durations else 0
        ),
        "minimum_scene_duration_seconds": (
            min(durations) if durations else 0
        ),
        "maximum_scene_duration_seconds": (
            max(durations) if durations else 0
        ),
        "detection_time_seconds": detection_time,
        "threshold": THRESHOLD
    }   

    report_path = os.path.join(
        output_folder,
        "segmentation_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Saved report: {report_path}")


def segment_all_videos(video_path=None):
    os.makedirs(SOURCE_DIR, exist_ok=True)

    if video_path:
        segment_video(video_path)
        return

    videos = [
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".mp4")
    ]

    if not videos:
        print(f"No .mp4 files found in '{SOURCE_DIR}'.")
        return

    for video in videos:
        segment_video(video)


if __name__ == "__main__":
    segment_all_videos()