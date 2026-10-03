import os
import sys

# Configure CUDA library path (libcublas.so.12 / libcudnn) on Linux for CTranslate2 / faster-whisper
if sys.platform.startswith("linux") and os.environ.get("_MMHD_CUDA_CONFIGURED") != "1":
    _cuda_dirs = []
    for _pkg in ["nvidia.cublas.lib", "nvidia.cudnn.lib"]:
        try:
            _mod = __import__(_pkg, fromlist=["__file__"])
            _cuda_dirs.append(os.path.dirname(_mod.__file__))
        except Exception:
            pass
    try:
        import site
        for _sp in site.getsitepackages():
            _nvidia_dir = os.path.join(_sp, "nvidia")
            if os.path.isdir(_nvidia_dir):
                for _child in os.listdir(_nvidia_dir):
                    _lib_p = os.path.join(_nvidia_dir, _child, "lib")
                    if os.path.isdir(_lib_p) and _lib_p not in _cuda_dirs:
                        _cuda_dirs.append(_lib_p)
    except Exception:
        pass

    if _cuda_dirs:
        _cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        _missing = [d for d in _cuda_dirs if d not in _cur_ld.split(":")]
        if _missing:
            _new_ld = ":".join(_missing + ([_cur_ld] if _cur_ld else []))
            _env = os.environ.copy()
            _env["LD_LIBRARY_PATH"] = _new_ld
            _env["_MMHD_CUDA_CONFIGURED"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)

# Ensure ~/.local/bin is on PATH for non-root environments (e.g. AIKosh)
_local_bin = os.path.expanduser("~/.local/bin")
if os.path.isdir(_local_bin) and _local_bin not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = f"{_local_bin}:{os.environ.get('PATH', '')}"

import json
import time
import csv
from datetime import datetime

# Ensure src/ is on the Python module search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from download import download_video
from segment import segment_all_videos
from candidate import process_video_candidates
from extract import extract_video_dataset

CSV_LOG_FILE = "batch_log.csv"
MD_LOG_FILE = "Batch-Log.md"


def parse_batch_input_file(filepath):
    """
    Parses a batch input file.
    Supports header directives:
      # Batch: Batch-01
      # Category: Stand-up Comedy
      # Notes: Initial test run
    And inline comments after URLs:
      https://youtube.com/watch?v=xyz # Good crowd reaction
    """
    batch_info = {
        "batch_id": os.path.splitext(os.path.basename(filepath))[0].replace("_", "-").title(),
        "category": "Mixed / General",
        "notes": "",
        "items": []  # list of {"url": url, "comment": comment}
    }

    if not os.path.exists(filepath):
        print(f"Batch file not found: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            # Check header directives
            if line_str.startswith("#"):
                lower = line_str.lower()
                if lower.startswith("# batch:") or lower.startswith("# batch_id:"):
                    batch_info["batch_id"] = line_str.split(":", 1)[1].strip()
                elif lower.startswith("# category:"):
                    batch_info["category"] = line_str.split(":", 1)[1].strip()
                elif lower.startswith("# notes:") or lower.startswith("# comment:"):
                    batch_info["notes"] = line_str.split(":", 1)[1].strip()
                continue

            # Extract URL and optional inline comment
            if "#" in line_str:
                url_part, comment_part = line_str.split("#", 1)
                url = url_part.strip()
                comment = comment_part.strip()
            else:
                url = line_str
                comment = ""

            if url.startswith("http://") or url.startswith("https://"):
                batch_info["items"].append({"url": url, "comment": comment})

    return batch_info


def read_video_stats(video_name):
    """Gathers scene, candidate, accepted, and rejection stats from output JSONs."""
    folder = os.path.join("sample_clips", video_name)
    stats = {
        "scenes": 0,
        "candidates": 0,
        "accepted": 0,
        "rejected": 0
    }

    seg_file = os.path.join(folder, "segmentation_report.json")
    if os.path.exists(seg_file):
        try:
            with open(seg_file, "r", encoding="utf-8") as f:
                stats["scenes"] = json.load(f).get("scene_count", 0)
        except Exception:
            pass

    cand_file = os.path.join(folder, "candidate_report.json")
    if os.path.exists(cand_file):
        try:
            with open(cand_file, "r", encoding="utf-8") as f:
                stats["candidates"] = json.load(f).get("candidate_count", 0)
        except Exception:
            pass

    ds_file = os.path.join(folder, "dataset.json")
    if os.path.exists(ds_file):
        try:
            with open(ds_file, "r", encoding="utf-8") as f:
                stats["accepted"] = len(json.load(f))
        except Exception:
            pass

    rej_file = os.path.join(folder, "diversity_rejections.json")
    if os.path.exists(rej_file):
        try:
            with open(rej_file, "r", encoding="utf-8") as f:
                stats["rejected"] = len(json.load(f))
        except Exception:
            pass

    return stats


def log_batch_results(batch_info, video_results, total_time_sec, val_status):
    """Logs batch results to both CSV and Markdown files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    tot_scenes = sum(r["scenes"] for r in video_results)
    tot_candidates = sum(r["candidates"] for r in video_results)
    tot_accepted = sum(r["accepted"] for r in video_results)
    tot_rejected = sum(r["rejected"] for r in video_results)

    # 1. Update CSV Log
    csv_exists = os.path.exists(CSV_LOG_FILE)
    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "Timestamp", "Batch_ID", "Category", "Video_ID", "Video_URL",
                "Scenes", "Candidates", "Accepted", "Rejected", "Validation",
                "Elapsed_Sec", "Comments"
            ])
        for r in video_results:
            writer.writerow([
                timestamp, batch_info["batch_id"], batch_info["category"],
                r["video_id"], r["url"], r["scenes"], r["candidates"],
                r["accepted"], r["rejected"], val_status,
                f"{r['elapsed']:.1f}", r["comment"]
            ])

    # 2. Update Markdown Log
    md_exists = os.path.exists(MD_LOG_FILE)
    with open(MD_LOG_FILE, "a", encoding="utf-8") as f:
        if not md_exists:
            f.write("# MMHD Phase 1: Batch Execution Log\n\n")
            f.write("This file tracks batch runs, video yields, validation status, and researcher notes.\n\n")
            f.write("| Date | Batch ID | Category | Videos | Candidates | Accepted | Rejections | Status | Duration | Batch Notes |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        duration_fmt = f"{int(total_time_sec // 60)}m {int(total_time_sec % 60)}s"
        notes_str = batch_info["notes"].replace("|", "/") if batch_info["notes"] else "None"
        
        f.write(f"| {date_str} | **{batch_info['batch_id']}** | {batch_info['category']} | "
                f"{len(video_results)} | {tot_candidates} | {tot_accepted} | {tot_rejected} | "
                f"`{val_status}` | {duration_fmt} | {notes_str} |\n")

        # Append detailed video breakdown block
        f.write(f"\n<details><summary><b>Detailed Breakdown: {batch_info['batch_id']}</b></summary>\n\n")
        f.write("| Video ID | Scenes | Candidates | Accepted | Rejected | Duration | Comment |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in video_results:
            f.write(f"| `{r['video_id']}` | {r['scenes']} | {r['candidates']} | {r['accepted']} | "
                    f"{r['rejected']} | {r['elapsed']:.1f}s | {r['comment'] or '-'} |\n")
        f.write("\n</details>\n\n---\n\n")

    print(f"\n[LOGGER] Batch logged to {CSV_LOG_FILE} and {MD_LOG_FILE}")


def run_validation():
    """Runs validate_dataset.py and captures the PASS/FAIL outcome."""
    try:
        import validate_dataset
        return "PASS"
    except SystemExit as e:
        if str(e) == "0" or not str(e):
            return "PASS"
        return "FAIL"
    except Exception as e:
        print(f"Validation exception: {e}")
        return "FAIL"


def main():
    target_file = "urls.txt"
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    batch_info = parse_batch_input_file(target_file)
    if not batch_info or not batch_info["items"]:
        print(f"No valid URLs found in {target_file}.")
        return

    print("=" * 65)
    print(f"BATCH ID  : {batch_info['batch_id']}")
    print(f"CATEGORY  : {batch_info['category']}")
    print(f"NOTES     : {batch_info['notes'] or 'None'}")
    print(f"VIDEOS    : {len(batch_info['items'])} queued")
    print("=" * 65)

    batch_start_time = time.time()
    video_results = []

    for idx, item in enumerate(batch_info["items"], start=1):
        url = item["url"]
        comment = item["comment"]
        print("\n" + "-" * 60)
        print(f"[{idx}/{len(batch_info['items'])}] VIDEO: {url}")
        if comment:
            print(f"Comment : {comment}")
        print("-" * 60)

        v_start = time.time()
        video_name = "unknown"

        try:
            # 1. Download
            video_path = download_video(url)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_folder = os.path.join("sample_clips", video_name)

            # 2. Segment
            segment_all_videos(video_path)

            # 3. Candidate Formation
            process_video_candidates(video_folder)

            # 4. Multimodal Extraction
            extract_video_dataset(video_folder)

            v_elapsed = time.time() - v_start
            stats = read_video_stats(video_name)

            video_results.append({
                "video_id": video_name,
                "url": url,
                "comment": comment,
                "scenes": stats["scenes"],
                "candidates": stats["candidates"],
                "accepted": stats["accepted"],
                "rejected": stats["rejected"],
                "elapsed": v_elapsed
            })

        except Exception as e:
            print(f"\n[ERROR] Processing failed for {url}: {e}")
            video_results.append({
                "video_id": video_name,
                "url": url,
                "comment": f"FAILED: {e}",
                "scenes": 0,
                "candidates": 0,
                "accepted": 0,
                "rejected": 0,
                "elapsed": time.time() - v_start
            })

    total_batch_sec = time.time() - batch_start_time

    # Run validation check
    print("\n" + "=" * 65)
    print("RUNNING POST-BATCH DATASET VALIDATION...")
    print("=" * 65)
    val_status = run_validation()

    # Log to CSV and Markdown
    log_batch_results(batch_info, video_results, total_batch_sec, val_status)

    # Print Project-Lead Handoff Report (MMHD Instructions Section 58)
    tot_cand = sum(r["candidates"] for r in video_results)
    tot_acc = sum(r["accepted"] for r in video_results)
    tot_rej = sum(r["rejected"] for r in video_results)

    print("\n" + "=" * 65)
    print(f"REPORT FOR PROJECT LEAD (BATCH: {batch_info['batch_id']})")
    print("=" * 65)
    print(f"Batch completed     : {batch_info['batch_id']}")
    print(f"Category            : {batch_info['category']}")
    print(f"Videos processed    : {len(video_results)}")
    print(f"Candidates generated: {tot_cand}")
    print(f"Accepted samples    : {tot_acc}")
    print(f"Diversity rejections: {tot_rej}")
    print(f"Validation          : {val_status}")
    print(f"Processing time     : {int(total_batch_sec // 60)}m {int(total_batch_sec % 60)}s")
    print(f"Notes               : {batch_info['notes'] or 'Standard batch execution'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
