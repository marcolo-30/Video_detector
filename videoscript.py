import asyncio
from concurrent.futures import ThreadPoolExecutor

import os
import argparse
import cv2
from ultralytics import YOLO
import time
import csv
import signal
import logging
from datetime import datetime
from pathlib import Path

from UDPDataSender import send_data_with_headers
from TCPDataSender import TCPClient

# =========================
# Local Telemetry (CSV)
# =========================
class LocalTelemetry:
    """Tiny drop-in that mimics telemetry.pushMetric -> writes to CSV locally."""
    def __init__(self, metrics_csv_path: str):
        self.metrics_csv_path = metrics_csv_path
        # Ensure header exists (idempotent)
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
        if not os.path.exists(metrics_csv_path) or os.path.getsize(metrics_csv_path) == 0:
            with open(metrics_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["ts", "name", "type", "value"])

    def pushMetric(self, name: str, mtype: str, value):
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.utcnow().isoformat(), name, mtype, value])

# =========================
# Globals / Config
# =========================
tcp_client = None
MAX_WORKERS = 6
QUEUE_SIZE = 50
frame_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

GLOBAL_PACKET_ID = 0
GLOBAL_RUNNING_STATE = True

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# annotated video handling
VIDEO_FPS = 30.0
video_writer = None
VIDEO_NAME_PREFIX = "video"  # will be set per input (e.g., "V1")

# logging will be configured inside main() once we know --output
logger = logging.getLogger("detector")
logger.setLevel(logging.INFO)

# =========================
# Signal handling (Windows-safe)
# =========================
def handle_signal(signum, frame):
    global GLOBAL_RUNNING_STATE
    print(f"Received signal: {signum}. Shutting down gracefully...")
    GLOBAL_RUNNING_STATE = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
if hasattr(signal, "SIGHUP"):  # not on Windows
    signal.signal(signal.SIGHUP, handle_signal)

# =========================
# UDP/TCP helpers
# =========================
async def send_data_with_headers_async(host, port, buffer, frame_id, object_id, total_objects):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, send_data_with_headers, host, port, buffer, frame_id, object_id, total_objects
    )

async def send_image_via_tcp(tcp_client_local, image, frame_id, object_id=0, total_objects=0):
    global GLOBAL_PACKET_ID
    if tcp_client_local is None:
        return

    loop = asyncio.get_event_loop()
    ok, encoded_image = await loop.run_in_executor(None, cv2.imencode, '.jpg', image)
    if not ok:
        logger.warning(f"JPEG encoding failed for frame {frame_id}, object {object_id}")
        return

    buffer = encoded_image.tobytes()
    GLOBAL_PACKET_ID += 1

    try:
        result = tcp_client_local.send_data(buffer, frame_id, object_id, total_objects)
        if result != 1:
            logger.warning(f"Failed to send frame {frame_id}, object {object_id}/{total_objects}")
        else:
            logger.info(f"Frame {frame_id}, object {object_id}/{total_objects} sent successfully (TCP)")
    except Exception as e:
        logger.exception(f"Error sending frame {frame_id}, object {object_id}: {e}")

async def send_image_via_udp(image, frame_id, object_id=0, total_objects=0):
    global GLOBAL_PACKET_ID
    loop = asyncio.get_event_loop()
    ok, encoded_image = await loop.run_in_executor(None, cv2.imencode, '.jpg', image)
    if not ok:
        logger.warning(f"JPEG encoding failed for frame {frame_id}, object {object_id}")
        return

    buffer = encoded_image.tobytes()
    udp_ip = os.getenv("UDP_IP", "127.0.0.1")
    udp_port = int(os.getenv("UDP_PORT", "5005"))
    GLOBAL_PACKET_ID += 1

    try:
        result = await send_data_with_headers_async(udp_ip, udp_port, buffer, frame_id, object_id, total_objects)
        if result != 1:
            logger.warning(f"Failed to send frame {frame_id}, object {object_id}/{total_objects}")
        else:
            logger.info(f"Frame {frame_id}, object {object_id}/{total_objects} sent successfully (UDP)")
    except Exception as e:
        logger.exception(f"Error sending frame {frame_id}, object {object_id}: {e}")

# =========================
# VIDEO FILE reader
# =========================
async def read_video_frames(video_path):
    global VIDEO_FPS
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video file {video_path}")
        return

    # capture FPS if available
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        VIDEO_FPS = float(fps)

    logger.info(f"Opened video file: {video_path} (fps={VIDEO_FPS})")
    frame_idx = 0

    try:
        loop = asyncio.get_event_loop()
        while GLOBAL_RUNNING_STATE:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.info("End of video or read error.")
                break

            try:
                frame_queue.put_nowait((frame_idx, frame))
            except asyncio.QueueFull:
                await frame_queue.put((frame_idx, frame))

            frame_idx += 1
    finally:
        cap.release()
        logger.info("Video reading stopped.")

# =========================
# Core processing
# =========================
def _frame_timestamp_str(frame_idx: int, fps: float) -> str:
    """Return 'MM:SS.ff' timestamp string for a given frame index and fps."""
    if fps <= 0:
        fps = 30.0
    secs = frame_idx / fps
    mm = int(secs // 60)
    ss = secs % 60
    return f"{mm:02d}:{ss:05.2f}"

async def process_frame(
    model,
    frame_data,
    telemetry,
    forwardFrameUDP=False,
    forwardTcp=False,
    tcpClient=None,
    display=False,
    save_video_path="",
    save_crops_dir="",
    save_frames_by_count_dir="",
    max_object_bucket=50
):
    global video_writer, VIDEO_FPS, GLOBAL_RUNNING_STATE, VIDEO_NAME_PREFIX

    frame_idx, im0 = frame_data
    t0 = time.time()

    results = model(im0)

    # Boxes/classes
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    box_count = len(boxes)

    # ---- Save RAW frame grouped by object count (includes 0) ----
    if save_frames_by_count_dir:
        within_upper_bound = (max_object_bucket <= 0) or (box_count <= max_object_bucket)
        if within_upper_bound:  # save 0..max_object_buket, or all if max=0
            bucket_dir = os.path.join(save_frames_by_count_dir, f"{box_count}objects")
            os.makedirs(bucket_dir, exist_ok=True)
            out_path = os.path.join(bucket_dir, f"{VIDEO_NAME_PREFIX}_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, im0)

    # Annotated frame with Ultralytics helper
    annotated = results[0].plot()  # BGR np.array

    # --- Overlay: timestamp, frame, boxes ---
    timestamp_str = _frame_timestamp_str(frame_idx, VIDEO_FPS)
    overlay_lines = [
        f"time {timestamp_str}",
        f"frame {frame_idx}",
        f"boxes {box_count}",
    ]
    y = 30
    for line in overlay_lines:
        cv2.putText(
            annotated, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 32

    # --- Visualize live ---
    if display:
        cv2.imshow("Detections", annotated)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            GLOBAL_RUNNING_STATE = False

    # --- Save annotated video lazily ---
    if save_video_path:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = annotated.shape[:2]
            video_writer = cv2.VideoWriter(save_video_path, fourcc, VIDEO_FPS, (w, h))
        video_writer.write(annotated)

    # --- Optional forwarding of crops ---
    tasks = []
    for i, (box, cls) in enumerate(zip(boxes, clss), start=1):
        x1, y1, x2, y2 = map(int, box)
        cropped_object = im0[y1:y2, x1:x2]
        if forwardFrameUDP:
            tasks.append(send_image_via_udp(cropped_object, frame_idx, i, box_count))
        if forwardTcp:
            tasks.append(send_image_via_tcp(tcpClient, cropped_object, frame_idx, i, box_count))

    if tasks:
        await asyncio.gather(*tasks)

    frame_latency_ms = (time.time() - t0) * 1000.0

    # Local metrics
    telemetry.pushMetric("frames_sent", "async_counter", frame_idx)
    telemetry.pushMetric("frame_latency", "gauge", round(frame_latency_ms, 3))
    telemetry.pushMetric("box_count", "gauge", box_count)

    logger.info(
        f"Processed frame {frame_idx} (t={timestamp_str}) | boxes={box_count} | latency_ms={frame_latency_ms:.2f}"
    )

async def process_stream(model, telemetry, forwardFrameUdp, forwardFrameTcp, tcpClient=None,
                         display=False, save_video_path="", save_crops_dir="",
                         save_frames_by_count_dir="", max_object_bucket=50):
    while GLOBAL_RUNNING_STATE:
        telemetry.pushMetric("detector_queue_size", "gauge", frame_queue.qsize())

        if frame_queue.empty():
            await asyncio.sleep(0.01)
            continue

        frame_data = await frame_queue.get()
        try:
            await process_frame(
                model,
                frame_data,
                telemetry,
                forwardFrameUDP=forwardFrameUdp,
                forwardTcp=forwardFrameTcp,
                tcpClient=tcpClient,
                display=display,
                save_video_path=save_video_path,
                save_crops_dir=save_crops_dir,
                save_frames_by_count_dir=save_frames_by_count_dir,
                max_object_bucket=max_object_bucket,
            )
        except Exception as e:
            logger.exception(f"Error processing frame {frame_data[0]}: {e}")

# =========================
# Images
# =========================
async def process_single_image(image_path, frame_idx, model, telemetry, udpForwardFrame, tcpForwardFrame,
                               tcpClient=None, display=False, save_crops_dir="",
                               save_frames_by_count_dir="", max_object_bucket=50):
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, cv2.imread, image_path)
    if img is None:
        logger.warning(f"Failed to load image: {image_path}")
        return
    await process_frame(
        model,
        (frame_idx, img),
        telemetry,
        udpForwardFrame,
        tcpForwardFrame,
        tcpClient,
        display=display,
        save_video_path="",   # not used in images mode
        save_crops_dir=save_crops_dir,
        save_frames_by_count_dir=save_frames_by_count_dir,
        max_object_bucket=max_object_bucket,
    )

async def process_images_async(image_dir, model, telemetry, udpForwardFrame, tcpForwardFrame, tcpClient=None,
                               display=False, save_crops_dir="", save_frames_by_count_dir="", max_object_bucket=50):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        logger.warning(f"No valid image files found in: {image_dir}")
        return

    logger.info(f"Found {len(files)} images. Processing asynchronously...")
    tasks = [
        process_single_image(
            os.path.join(image_dir, f), i, model, telemetry,
            udpForwardFrame, tcpForwardFrame, tcpClient,
            display=display, save_crops_dir=save_crops_dir,
            save_frames_by_count_dir=save_frames_by_count_dir,
            max_object_bucket=max_object_bucket,
        )
        for i, f in enumerate(files)
    ]
    await asyncio.gather(*tasks)
    logger.info("Finished processing all images.")

# =========================
# Per-video runner
# =========================
async def run_single_video(video_path, args, telemetry):
    """Run the full pipeline for a single video path."""
    global VIDEO_FPS, video_writer, GLOBAL_RUNNING_STATE, frame_queue, VIDEO_NAME_PREFIX

    # Reset per-video globals
    VIDEO_FPS = 30.0
    video_writer = None
    GLOBAL_RUNNING_STATE = True
    frame_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    # Per-video name prefix (used for saved frames so they don't overwrite)
    VIDEO_NAME_PREFIX = os.path.splitext(os.path.basename(video_path))[0]

    # Compute per-video annotated output path
    save_video_path = args.save_video
    if save_video_path:
        svp = Path(save_video_path)
        if svp.is_dir() or str(save_video_path).endswith(("/", "\\", os.path.sep)):
            save_video_path = str(svp / f"{VIDEO_NAME_PREFIX}_out.mp4")
        elif not svp.suffix:  # no extension -> treat like dir
            svp.mkdir(parents=True, exist_ok=True)
            save_video_path = str(svp / f"{VIDEO_NAME_PREFIX}_out.mp4")
        else:
            svp.parent.mkdir(parents=True, exist_ok=True)

    # Ensure frames base dir exists if requested
    if args.save_frames_by_count:
        Path(args.save_frames_by_count).mkdir(parents=True, exist_ok=True)

    # Optional TCP client per video
    tcp_client_local = None
    if args.tcp_forward:
        tcp_ip = os.getenv("TCP_IP", "127.0.0.1")
        tcp_port = int(os.getenv("TCP_PORT", "5005"))
        tcp_client_local = TCPClient(tcp_ip, tcp_port)
        tcp_client_local.connect()
        logger.info(f"TCP client connected to {tcp_ip}:{tcp_port}")

    # Create model per video (keeps things simple/isolated)
    model = YOLO(args.model)

    await asyncio.gather(
        read_video_frames(video_path),
        process_stream(
            model=model,
            telemetry=telemetry,
            forwardFrameUdp=args.udp_forward,
            forwardFrameTcp=args.tcp_forward,
            tcpClient=tcp_client_local,
            display=args.display,
            save_video_path=save_video_path,
            save_crops_dir=args.save_crops,
            save_frames_by_count_dir=args.save_frames_by_count,
            max_object_bucket=args.max_object_bucket,
        ),
    )

    # Cleanup per-video visuals
    try:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
    except Exception:
        pass

# =========================
# Main
# =========================
async def main():
    parser = argparse.ArgumentParser(description="Local-video/image object detector with local telemetry")
    parser.add_argument("--mode", choices=["video", "images", "none"], required=True,
                        help="Input mode: 'video' for a local video file, 'images' for a directory of images.")
    parser.add_argument("--input", required=False,
                        help="Path to a video file (for --mode video) or a directory (for --mode images).")

    # NEW: process a whole folder of videos sequentially
    parser.add_argument("--video-dir", default="",
                        help="If set (and --mode video), process all videos in this directory sequentially.")

    parser.add_argument("--output", required=False, default="./output",
                        help="Output directory for logs and metrics (default: ./output)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Path to the YOLO weights (default: yolo11n.pt).")
    parser.add_argument("--udp-forward", action="store_true", help="Forward cropped objects via UDP")
    parser.add_argument("--tcp-forward", action="store_true", help="Forward cropped objects via TCP")

    # visualization/persistence controls
    parser.add_argument("--display", action="store_true",
                        help="Show a live window with annotated detections")
    parser.add_argument("--save-video", default="",
                        help=("If a file path, saves annotated video there; "
                              "if a directory or ends with '/', saves as <VideoName>_out.mp4 inside it; "
                              "empty = don't save"))
    parser.add_argument("--save-crops", default="",
                        help="(Optional) Directory to save cropped objects to forward")

    # save frames grouped by object count (includes 0)
    parser.add_argument("--save-frames-by-count", default="",
        help="Base dir to save RAW frames grouped by detected object count "
             "(e.g., base/2objects/<VideoName>_frame_000123.jpg). Empty = don't save.")
    parser.add_argument("--max-object-bucket", type=int, default=50,
        help="Only save when boxes <= this value (default 50). Use 0 to save for any count.")

    args = parser.parse_args()

    # Paths & logging
    os.makedirs(args.output, exist_ok=True)
    app_log_path = os.path.join(args.output, "app.log")
    metrics_csv_path = os.path.join(args.output, "metrics.csv")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(app_log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    telemetry = LocalTelemetry(metrics_csv_path)
    logger.info(f"App log: {app_log_path}")
    logger.info(f"Metrics CSV: {metrics_csv_path}")

    # MODE DISPATCH
    if args.mode == "video" and args.video_dir:
        # Process all video files in the folder (common video extensions)
        exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
        folder = Path(args.video_dir)
        if not folder.is_dir():
            logger.error(f"--video-dir not found or not a directory: {folder}")
            return

        video_files = sorted([str(p) for p in folder.iterdir() if p.suffix.lower() in exts])
        if not video_files:
            logger.error(f"No video files found in {folder}")
            return

        logger.info(f"Found {len(video_files)} videos in {folder}. Starting sequential processing...")
        for idx, vp in enumerate(video_files, 1):
            logger.info(f"[{idx}/{len(video_files)}] Processing {vp}")
            try:
                await run_single_video(vp, args, telemetry)
            except Exception as e:
                logger.exception(f"Error while processing {vp}: {e}")
        logger.info("All videos processed.")
        return

    # Single inputs (original behavior)
    if args.mode == "video":
        if not args.input:
            logger.error("Error: --input path to a video file is required for --mode video (or use --video-dir).")
            return
        await run_single_video(args.input, args, telemetry)

    elif args.mode == "images":
        if not args.input or not os.path.isdir(args.input):
            logger.error(f"Error: Input directory does not exist: {args.input}")
            return
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            logger.error(f"No images found in {args.input}")
            return
        global VIDEO_NAME_PREFIX, frame_queue
        VIDEO_NAME_PREFIX = os.path.basename(os.path.normpath(args.input))
        frame_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        await process_images_async(
            args.input, YOLO(args.model), telemetry,
            args.udp_forward, args.tcp_forward, tcpClient=None,
            display=args.display,
            save_crops_dir=args.save_crops,
            save_frames_by_count_dir=args.save_frames_by_count,
            max_object_bucket=args.max_object_bucket,
        )

    else:  # "none"
        while True:
            telemetry.pushMetric("heartbeat", "counter", 1)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(1)
