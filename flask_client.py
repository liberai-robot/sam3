"""Client for flask_server.py: send an .mp4 + text prompt; save mask frames
and a mask .mp4.

After a successful run, --out-dir contains:
    <out-dir>/frames/<NNNNN>.png   - per-frame binary mask PNGs
    <out-dir>/mask.mp4             - mask video at the input video's fps

Example:
    python flask_client.py --video clip.mp4 --text "hand" --out-dir ./out
    python flask_client.py --server http://gpu-host:8765 \
        --video ./clip.mp4 --text "left hand" --out-dir ./out
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import requests


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8765",
                    help="base URL of flask_server.py")
    ap.add_argument("--video", required=True, type=Path,
                    help="input video file (.mp4/.mov/.mkv/...)")
    ap.add_argument("--text", required=True,
                    help="text prompt (e.g. 'left hand', 'cup', ...)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output dir; receives frames/<NNNNN>.png + mask.mp4")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-request timeout in seconds (default: no timeout)")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[client] video not found: {args.video}", file=sys.stderr)
        sys.exit(2)
    if args.video.suffix.lower() not in VIDEO_EXTS:
        print(f"[client] unsupported video extension {args.video.suffix!r}; "
              f"allowed: {sorted(VIDEO_EXTS)}", file=sys.stderr)
        sys.exit(2)

    try:
        h = requests.get(f"{args.server}/health", timeout=5).json()
        print(f"[client] server health: {h}")
    except Exception as e:
        print(f"[client] server health probe failed: {e}", file=sys.stderr)
        sys.exit(3)

    size_mb = args.video.stat().st_size / 1e6
    print(f"[client] uploading video {args.video.name} ({size_mb:.1f} MB) "
          f"text={args.text!r}")

    t0 = time.time()
    with open(args.video, "rb") as fh:
        r = requests.post(
            f"{args.server}/segment",
            files={"video": (args.video.name, fh, "video/mp4")},
            data={"text": args.text},
            timeout=args.timeout,
            stream=True,
        )

    if r.status_code != 200:
        print(f"[client] HTTP {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)

    out_buf = io.BytesIO()
    n_bytes = 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        if not chunk:
            continue
        out_buf.write(chunk)
        n_bytes += len(chunk)
    out_buf.seek(0)
    dt = time.time() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_root = args.out_dir.resolve()
    n_png = 0
    n_mp4 = 0
    with zipfile.ZipFile(out_buf) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (args.out_dir / info.filename).resolve()
            # Defend against zip-slip from a hostile/buggy server.
            try:
                target.relative_to(out_root)
            except ValueError:
                raise RuntimeError(f"zip entry escapes out-dir: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            ext = target.suffix.lower()
            if ext == ".png":
                n_png += 1
            elif ext == ".mp4":
                n_mp4 += 1

    print(f"[client] saved {n_png} mask PNGs + {n_mp4} mask video "
          f"({n_bytes / 1e6:.1f} MB zip) → {args.out_dir}")
    print(
        f"[client]   frames: {r.headers.get('X-Frames')}  "
        f"size: {r.headers.get('X-Width')}x{r.headers.get('X-Height')}  "
        f"fps: {r.headers.get('X-FPS')}"
    )
    print(
        f"[client]   server inference: {r.headers.get('X-Inference-Seconds')}s  "
        f"total wall: {dt:.1f}s"
    )


if __name__ == "__main__":
    main()
