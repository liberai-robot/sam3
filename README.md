# SAM3 Flask Server / Client

A thin HTTP wrapper around Meta's [SAM 3](./ORIGINAL_README.md) text-prompted
video segmentation model. Run the GPU-heavy predictor once on a remote machine
(`flask_server.py`), then call it from any laptop on the same LAN
(`flask_client.py`) with an **`.mp4` file plus a text prompt** — get back a
folder of binary mask PNGs **and** a mask `.mp4` video (encoded at the input
video's fps).

For everything about the SAM 3 model itself (architecture, checkpoints,
benchmarks, license), see [`ORIGINAL_README.md`](./ORIGINAL_README.md).

## How it works

```
   laptop                                  GPU server
┌──────────────┐    POST /segment       ┌──────────────────────────┐
│ video.mp4    │ ─── *.mp4 + text ───►  │ SAM3 predictor (loaded   │
│ "left hand"  │                        │ once at startup, kept    │
│              │ ◄──── masks.zip ────── │ resident on GPU)         │
│ out/         │     (frames/ +         │                          │
│  ├ frames/   │      mask.mp4)         │                          │
│  └ mask.mp4  │                        │                          │
└──────────────┘                        └──────────────────────────┘
```

- The server loads the SAM 3 predictor **once at startup** and keeps it on the
  GPU between requests — there's no per-request model load cost.
- Requests are serialized with a lock (single GPU, single session at a time).
- The client uploads the `.mp4` directly (no client-side transcoding),
  streams a zip back, and extracts it into `--out-dir`, producing a
  `frames/` subfolder of mask PNGs alongside a single `mask.mp4`.

## 1. Server setup (on the GPU machine)

### Install SAM 3

Follow the installation steps in [`ORIGINAL_README.md`](./ORIGINAL_README.md#installation):
create the `sam3` conda/mamba env, install PyTorch with CUDA, `pip install -e .`,
and authenticate with Hugging Face to download the checkpoints.

### Install Flask

```bash
mamba activate sam3
pip install flask
```

### Run the server

```bash
mamba activate sam3
cd /path/to/sam3
python flask_server.py
```

> **Note on layout.** If your `flask_server.py` lives in a directory that also
> contains the cloned upstream `sam3/` repo (e.g.
> `foundation_models/sam3/{flask_server.py, sam3/}`), Python's default import
> machinery would resolve `import sam3` to that sibling directory as an empty
> *namespace package* and shadow the pip-installed package — causing
> `pkg_resources` to fail with `TypeError: expected str ... not NoneType`.
> The server strips its own directory from `sys.path` at startup to avoid
> this, so the layout above just works.

First startup downloads/loads the SAM 3 checkpoint (takes a minute or two); you
should see:

```
[sam3] loading predictor (version=sam3, fa3=False)...
[sam3] predictor ready in 42.3s (cuda=True)
[sam3] listening on http://0.0.0.0:8765 (max upload 2048 MB)
```

The server binds to `0.0.0.0` by default, so any machine on the LAN can reach
it at `http://<server-ip>:8765`.

### Environment variables

| Variable        | Default  | Meaning                                       |
| --------------- | -------- | --------------------------------------------- |
| `HOST`          | `0.0.0.0`| Bind address                                  |
| `PORT`          | `8765`   | Bind port                                     |
| `SAM3_VERSION`  | `sam3`   | `sam3` or `sam3.1` (see ORIGINAL_README)      |
| `MAX_UPLOAD_MB` | `2048`   | Max request body size (uploaded video file)   |

Example with overrides:

```bash
PORT=9000 SAM3_VERSION=sam3.1 MAX_UPLOAD_MB=4096 python flask_server.py
```

### Keep it running

For a quick session, just leave it in a terminal (or a `tmux`/`screen`
window). For longer-lived use, run it under `systemd`, `supervisord`, or
similar — the server has no internal restart logic.

## 2. Client setup (on your laptop)

Only `requests` is needed — no SAM 3 install, no GPU.

```bash
pip install requests
```

Copy `flask_client.py` to the laptop (or check out this repo on the laptop).

### Sanity-check the server

```bash
curl http://<server-ip>:8765/health
# {"cuda":true,"ok":true,"version":"sam3"}
```

`ok: true` means the predictor finished loading and is ready to accept
`/segment` calls.

### Run a segmentation

```bash
python flask_client.py \
    --server  http://<server-ip>:8765 \
    --video   /path/to/clip.mp4 \
    --text    "left hand" \
    --out-dir /path/to/out
```

Allowed extensions: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`. The video
is uploaded directly (no client-side transcoding). The text prompt is applied
at `frame_index=0` and propagated through the whole sequence.

After a successful run, `--out-dir` contains:

```
out/
├── frames/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── mask.mp4
```

- `frames/<NNNNN>.png` — per-frame binary mask PNGs. uint8 with `0` for
  background and `255` for the segmented region (union over all instances SAM 3
  returns for that frame). Filenames are zero-padded frame indices with width
  `max(5, len(str(num_frames - 1)))`.
- `mask.mp4` — mask video encoded with `mp4v` at the **input video's fps**
  (falls back to 30 if the input doesn't report a valid fps). Good for quick
  preview/playback; the PNGs are the authoritative output (no lossy
  re-encoding).

### Client flags

| Flag           | Required | Default                  | Meaning                                |
| -------------- | -------- | ------------------------ | -------------------------------------- |
| `--server`     | no       | `http://127.0.0.1:8765`  | Server base URL                        |
| `--video`      | yes      | —                        | Input video file (`.mp4`/`.mov`/...)   |
| `--text`       | yes      | —                        | Prompt, e.g. `"left hand"`, `"cup"`    |
| `--out-dir`    | yes      | —                        | Receives `frames/*.png` + `mask.mp4`   |
| `--timeout`    | no       | none                     | Per-request timeout in seconds         |

## 3. HTTP API

If you'd rather call the server from another language, here's the wire format.

### `GET /health`

```json
{"ok": true, "version": "sam3", "cuda": true}
```

### `POST /segment`

`multipart/form-data`:

| Field    | Type                  | Required | Notes                                                |
| -------- | --------------------- | -------- | ---------------------------------------------------- |
| `video`  | file (`.mp4`/...)     | yes      | Single video file (`.mp4`/`.mov`/`.mkv`/...)         |
| `text`   | string                | yes      | Prompt, propagated from `frame_index=0`              |

Response: `application/zip` whose entries are:

```
frames/<NNNNN>.png   - per-frame binary mask PNGs (0/255)
mask.mp4             - mask video encoded at the input video's fps
```

Plus these headers:

| Header                 | Meaning                                       |
| ---------------------- | --------------------------------------------- |
| `X-Frames`             | Number of mask frames returned                |
| `X-Width`, `X-Height`  | Frame resolution                              |
| `X-FPS`                | fps used to encode `mask.mp4` (= input fps)   |
| `X-Inference-Seconds`  | Server-side inference wall time               |

Error responses are JSON (`400` for bad input, `500` for inference failure,
`503` if the predictor hasn't finished loading).

## Tips & gotchas

- **First request is not slower than later ones** — model weights are loaded
  at server startup, not on the first call.
- **One request at a time.** A `threading.Lock` serializes inference; multiple
  concurrent clients will queue, not parallelize.
- **Mask video is for preview.** `mask.mp4` is encoded with `mp4v` (lossy) at
  the input video's fps. For pixel-accurate downstream use, work from the PNGs
  in `frames/`.
- **Upload size.** Default cap is 2 GB. Bump `MAX_UPLOAD_MB` if your video is
  larger.
- **LAN, not the public internet.** There's no auth on these endpoints — only
  expose the port inside a trusted network (or front it with a reverse proxy
  that adds auth/TLS).
- **GPU memory** is released back via `torch.cuda.empty_cache()` after each
  request, but the model itself stays resident.

## See also

- [`ORIGINAL_README.md`](./ORIGINAL_README.md) — upstream SAM 3 README:
  installation prerequisites, checkpoint access, benchmarks, license, and
  citation.
- [`flask_server.py`](./flask_server.py) — the server implementation.
- [`flask_client.py`](./flask_client.py) — the client implementation.
