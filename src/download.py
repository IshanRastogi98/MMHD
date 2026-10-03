import os
import re
import shutil
import yt_dlp

SOURCE_DIR = "source_videos"

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=dYp-KUK73RE",
]


def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from various URL formats or raw IDs."""
    url = url.strip()
    if len(url) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    match = re.search(r"(?:v=|\/|be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else ""


def find_cookie_file() -> str | None:
    """Locates a cookies.txt file in environment variables or common project locations."""
    env_cookie = os.environ.get("YTDLP_COOKIES")
    if env_cookie and os.path.isfile(env_cookie) and os.path.getsize(env_cookie) > 0:
        return env_cookie

    search_paths = [
        "cookies.txt",
        os.path.join("..", "cookies.txt"),
        os.path.join(os.path.dirname(__file__), "..", "cookies.txt"),
        os.path.expanduser("~/.config/yt-dlp/cookies.txt"),
    ]
    for p in search_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return os.path.abspath(p)
    return None


def find_ffmpeg_dir() -> str | None:
    """Finds FFmpeg executable directory for yt-dlp merging."""
    ff_path = shutil.which("ffmpeg")
    if ff_path:
        return os.path.dirname(ff_path)

    local_ff = os.path.expanduser("~/.local/bin/ffmpeg")
    if os.path.isfile(local_ff) and os.access(local_ff, os.X_OK):
        return os.path.dirname(local_ff)

    try:
        import imageio_ffmpeg
        return os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    return None


def download_video(url: str) -> str:
    """
    Downloads a video with automated bot-evasion, client fallback,
    and local-disk caching for Google Colab and AIKosh environments.
    """
    os.makedirs(SOURCE_DIR, exist_ok=True)

    # Fast path: Check if already downloaded on disk before hitting the network
    video_id = extract_video_id(url)
    if video_id:
        cached_path = os.path.join(SOURCE_DIR, f"{video_id}.mp4")
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1024:
            print(f"[yt-dlp] Video already exists on disk: {cached_path} (Skipping download)")
            return cached_path

    cookie_file = find_cookie_file()
    if cookie_file:
        print(f"[yt-dlp] Authenticating with cookie file: {cookie_file}")

    ffmpeg_dir = find_ffmpeg_dir()

    # Determine fallback client tiers to bypass datacenter IP bot detection
    if cookie_file:
        client_tiers = [
            ["web", "android"],
            ["android", "ios"],
        ]
    else:
        # Datacenter IPs (Colab, AIKosh) fail 'web' client bot check; mobile clients bypass it
        client_tiers = [
            ["android", "ios"],
            ["android"],
            ["mweb", "web_safari"],
            ["web"],
        ]

    # Detect available JS runtime (e.g. node in Colab)
    js_runtimes = {}
    if shutil.which("node"):
        js_runtimes["node"] = {}
    elif shutil.which("deno"):
        js_runtimes["deno"] = {}

    last_error = None

    for tier_idx, clients in enumerate(client_tiers):
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(SOURCE_DIR, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extractor_args": {
                "youtube": {
                    "player_client": clients
                }
            },
        }

        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file

        if ffmpeg_dir:
            ydl_opts["ffmpeg_location"] = ffmpeg_dir

        if js_runtimes:
            ydl_opts["js_runtimes"] = js_runtimes

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                actual_id = info.get("id") or video_id
                output_path = os.path.join(SOURCE_DIR, f"{actual_id}.mp4")

                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    print(f"[yt-dlp] Successfully downloaded: {actual_id} -> {output_path}")
                    return output_path
                else:
                    # In case yt-dlp downloaded with a different extension and merged
                    for f in os.listdir(SOURCE_DIR):
                        if f.startswith(actual_id) and f.endswith(".mp4"):
                            resolved = os.path.join(SOURCE_DIR, f)
                            print(f"[yt-dlp] Downloaded: {actual_id} -> {resolved}")
                            return resolved

        except Exception as e:
            err_msg = str(e)
            last_error = e
            # If bot challenge or client rejection, attempt next client tier
            if any(k in err_msg for k in ["Sign in to confirm", "bot", "HTTP Error 429", "Requested format is not available"]):
                if tier_idx < len(client_tiers) - 1:
                    next_clients = client_tiers[tier_idx + 1]
                    print(f"[yt-dlp] Client tier {clients} encountered bot/format limit. Falling back to {next_clients}...")
                    continue
            # If it's a non-recoverable error and not a bot issue, raise or continue
            if tier_idx == len(client_tiers) - 1:
                break

    raise RuntimeError(
        f"Failed to download {url} across all player clients. Last error: {last_error}\n"
        f"Tip: If YouTube has flagged this datacenter IP, export a 'cookies.txt' from your browser "
        f"and place it in the notebook directory."
    )


if __name__ == "__main__":
    for url in VIDEO_URLS:
        try:
            download_video(url)
        except Exception as e:
            print(f"Download failed: {url}")
            print(f"Error: {e}")