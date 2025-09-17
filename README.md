# Video Object Detection Pipeline

This script performs object detection on videos or images using **YOLO** from the `ultralytics` library.
It supports processing a single video, multiple videos in a directory, or a directory of images.

## Features

✅ Process a **single video** or **all videos in a folder sequentially**  
✅ Live visualization of detections with bounding boxes  
✅ Save annotated video (`<VideoName>_out.mp4`)  
✅ Save **raw frames grouped by object count** (including 0objects) with per-video prefixes  
✅ Log performance metrics (frame latency, box count, etc.) to CSV  
✅ Optional UDP/TCP forwarding of cropped objects

## Requirements

```bash
pip install ultralytics opencv-python
```

Optional: `UDPDataSender.py` and `TCPDataSender.py` should be present if you want to use `--udp-forward` or `--tcp-forward`.

## Usage

### Process a single video

```bash
python videoscript.py --mode video --input ./Videos/V2.mp4 --display --save-video ./output/V2_out.mp4 --save-frames-by-count ./output/frames
```

### Process a whole folder of videos

```bash
python videoscript.py --mode video --video-dir ./Videos --display --save-video ./output --save-frames-by-count ./output/frames --max-object-bucket 50
```

This will:
- Show a live window with bounding boxes
- Save annotated videos to `./output/<VideoName>_out.mp4`
- Save raw frames to folders like `./output/frames/0objects/VideoName_frame_000001.jpg`

### Process a directory of images

```bash
python videoscript.py --mode images --input ./images --save-frames-by-count ./output/frames
```

### Arguments

| Argument | Description |
|---------|-------------|
| `--mode` | Required. `video`, `images`, or `none` |
| `--input` | Path to a single video file or image directory |
| `--video-dir` | Path to a folder containing multiple videos to process sequentially |
| `--output` | Directory for logs and metrics (default: `./output`) |
| `--model` | YOLO model path (default: `yolo11n.pt`) |
| `--display` | Show live annotated detections |
| `--save-video` | Directory or file path to save annotated videos |
| `--save-frames-by-count` | Base dir to save frames grouped by object count |
| `--max-object-bucket` | Only save frames with `boxes <= N` (0 = save all) |
| `--udp-forward` | Forward cropped detections via UDP |
| `--tcp-forward` | Forward cropped detections via TCP |

### Output Example

```
output/
 ├── app.log
 ├── metrics.csv
 ├── V2_out.mp4
 └── frames/
      ├── 0objects/
      │    ├── V2_frame_000001.jpg
      │    └── V2_frame_000002.jpg
      ├── 1objects/
      │    └── V2_frame_000045.jpg
      └── 2objects/
           └── V2_frame_000123.jpg
```

### Stopping the Script

Press **`q`** or **`ESC`** in the display window to stop early.

---

Made for experimenting with YOLO-based detection, telemetry, and frame saving in batch processing workflows.


# Video From Images 

This repository contains a Python script to generate a video from a folder of images.  
Images can be repeated randomly across the video but never appear twice in a row, giving the impression of natural variation.

## Features
- **Custom video size** (e.g., 640×480 VGA)
- **Exact duration control** (e.g., 60 seconds)
- **Random selection with non-adjacent repeats**
- **Letterboxing** to preserve image aspect ratio
- **Deterministic mode** with `--seed` for reproducible results

## Installation

```bash
[git clone https://github.com/marcolo-30/Video_detector.git](https://github.com/marcolo-30/Video_detector.git)
cd Video_detector/videogen/
pip install opencv-python
```

## Usage

```bash
python make_video_from_images.py   --images-dir ./images   --output ./output.mp4   --fps 30   --duration 60   --shot-sec 0.5   --size 640x480   --mode random   --seed 42
```

### Parameters
| Option           | Description |
|------------------|-------------|
| `--images-dir`   | Folder containing images |
| `--output`       | Output video path |
| `--fps`          | Frames per second |
| `--duration`     | Total duration of video in seconds |
| `--shot-sec`     | Seconds each image stays before switching |
| `--size`         | Video size in `WxH` format |
| `--mode`         | `random` or `pattern` (pattern repeats a shuffled mini-sequence) |
| `--seed`         | Optional seed for reproducible results |

---

## Example

Generate a 640×480, 60-second video from `./frames`:

```bash
python make_video_from_images.py --images-dir ./frames --output video.mp4 --size 640x480
```

---

## License
MIT License – feel free to use and modify.
