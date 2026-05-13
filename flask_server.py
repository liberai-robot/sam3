"""SAM3 text-prompted segmentation server (.mp4 in, mask frames + mask .mp4 out).

POST /segment
    multipart/form-data:
        video : .mp4 (or other ffmpeg-readable) video file
        text  : prompt string (e.g. "hand", "left hand holding a cup")
    response:
        body  : .zip with this layout:
                    frames/<NNNNN>.png   - per-frame binary mask PNGs (0/255)
                    mask.mp4             - mask video, encoded at input fps
        headers: X-Frames, X-Inference-Seconds, X-Width, X-Height, X-FPS

GET  /health
    {ok, version, cuda}

Start:
    mamba activate sam3
    python flask_server.py
"""

import io
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

# If this script lives next to a "sam3/" repo-checkout dir (no top-level
# __init__.py), Python's default PathFinder resolves `import sam3` to that
# directory as a namespace package and shadows the pip-installed editable
# package (whose meta_path finder runs later). Strip the script's directory
# from sys.path so the editable install wins.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from sam3.model_builder import build_sam3_predictor


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
VERSION = os.environ.get("SAM3_VERSION", "sam3")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "2048"))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# SAM3 stream API isn't reentrant + single GPU → serialize.
_INFERENCE_LOCK = threading.Lock()
PREDICTOR = None


def _frame_union_mask(outputs: dict, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for m in outputs["out_binary_masks"]:
        if m.shape != (h, w):
            m = cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        mask |= m.astype(bool)
    return mask


def _run_segmentation(
    video_path: Path,
    text: str,
    h: int,
    w: int,
    on_mask,
) -> int:
    resp = PREDICTOR.handle_request(
        {"type": "start_session", "resource_path": str(video_path)}
    )
    session_id = resp["session_id"]

    try:
        PREDICTOR.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text,
            }
        )

        n = 0
        for response in PREDICTOR.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            fi = response["frame_index"]
            mask = _frame_union_mask(response["outputs"], h, w)
            mask_u8 = (mask.astype(np.uint8) * 255)
            on_mask(fi, mask_u8)
            n += 1
    finally:
        PREDICTOR.handle_request(
            {"type": "close_session", "session_id": session_id}
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n


def _probe_video(video_path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"video has invalid dimensions {w}x{h}: {video_path}")
    if not (fps > 0):
        fps = 30.0
    return n_frames, w, h, fps


def _encode_mask_video(frames_dir: Path, out_path: Path, w: int, h: int, fps: float) -> int:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter for mask.mp4 (codec mp4v)")
    n = 0
    try:
        for png in sorted(frames_dir.iterdir()):
            if png.suffix.lower() != ".png":
                continue
            img = cv2.imread(str(png))  # imread expands 1-ch PNG → BGR
            if img is None:
                continue
            writer.write(img)
            n += 1
    finally:
        writer.release()
    return n


def _zip_dir_to_bytes(src_dir: Path) -> bytes:
    # PNGs and .mp4 are already-compressed binaries — ZIP_STORED avoids
    # spending CPU on deflate that won't shrink them.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for fp in sorted(src_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=fp.relative_to(src_dir).as_posix())
    return buf.getvalue()


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": PREDICTOR is not None,
            "version": VERSION,
            "cuda": torch.cuda.is_available(),
        }
    )


@app.post("/segment")
def segment():
    if PREDICTOR is None:
        return jsonify({"error": "predictor not ready"}), 503
    if "video" not in request.files:
        return jsonify({"error": "missing multipart field 'video'"}), 400

    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing or empty field 'text'"}), 400

    up = request.files["video"]
    suffix = Path(up.filename or "in.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTS:
        return jsonify(
            {"error": f"unsupported video extension {suffix!r}; allowed: {sorted(VIDEO_EXTS)}"}
        ), 400

    with tempfile.TemporaryDirectory(prefix="sam3_") as tmp_str:
        tmp = Path(tmp_str)
        vid_path = tmp / f"in{suffix}"
        up.save(str(vid_path))

        try:
            n_total, w, h, fps = _probe_video(vid_path)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400

        # Response layout: out_root/frames/<NNNNN>.png + out_root/mask.mp4
        out_root = tmp / "out"
        frames_dir = out_root / "frames"
        frames_dir.mkdir(parents=True)
        mp4_out_path = out_root / "mask.mp4"

        pad = max(5, len(str(max(n_total - 1, 0))))

        def on_mask(fi, mask_u8):
            cv2.imwrite(str(frames_dir / f"{str(fi).zfill(pad)}.png"), mask_u8)

        t0 = time.perf_counter()
        try:
            # SAM3's predictor enters a bf16 autocast in __init__ on the main
            # thread, but autocast is thread-local — Flask worker threads don't
            # inherit it, leading to "BFloat16 vs Float" matmul mismatches.
            # Re-enter autocast here per-request.
            with _INFERENCE_LOCK, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                n_frames = _run_segmentation(vid_path, text, h, w, on_mask)
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500
        dt = time.perf_counter() - t0

        try:
            _encode_mask_video(frames_dir, mp4_out_path, w, h, fps)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        data = _zip_dir_to_bytes(out_root)

    resp = Response(data, mimetype="application/zip")
    resp.headers["Content-Disposition"] = 'attachment; filename="masks.zip"'
    resp.headers["X-Frames"] = str(n_frames)
    resp.headers["X-Width"] = str(w)
    resp.headers["X-Height"] = str(h)
    resp.headers["X-FPS"] = f"{fps:.3f}"
    resp.headers["X-Inference-Seconds"] = f"{dt:.3f}"
    return resp


def _init_predictor():
    global PREDICTOR
    print(f"[sam3] loading predictor (version={VERSION}, fa3=False)...", flush=True)
    t0 = time.perf_counter()
    PREDICTOR = build_sam3_predictor(version=VERSION, use_fa3=False)
    print(
        f"[sam3] predictor ready in {time.perf_counter() - t0:.1f}s "
        f"(cuda={torch.cuda.is_available()})",
        flush=True,
    )


if __name__ == "__main__":
    _init_predictor()
    print(
        f"[sam3] listening on http://{HOST}:{PORT} "
        f"(max upload {MAX_UPLOAD_MB} MB)",
        flush=True,
    )
    app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
