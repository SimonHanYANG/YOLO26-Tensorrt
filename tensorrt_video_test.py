"""
TensorRT 视频推理：检测 + 跟踪可视化
- 统一输出 1920x1200 分辨率
- 绿色 bbox，无文字
- 支持 ByteTrack/BotSort 轨迹绘制
- 统计推理时间

用法:
  python tensorrt_video_test.py --engine best.engine                     # 检测
  python tensorrt_video_test.py --engine best_fp16.engine                # 检测 FP16
  python tensorrt_video_test.py --engine best.engine --mode track        # 跟踪
  python tensorrt_video_test.py --engine best.engine --mode both         # 检测+跟踪都跑
"""

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"  # 静音 ffmpeg 警告

import argparse
import time
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

# ── 默认配置 ──────────────────────────────────────────────────────
VIDEO_PATH = "dataset/XiangYa-test-videoes/gray_video.mp4"
OUTPUT_DIR = "video_test_results"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMG_SIZE = 640
WARMUP_FRAMES = 5
DEVICE = 1

# 输出分辨率（所有帧统一 resize 到这个尺寸）
OUT_WIDTH = 1920
OUT_HEIGHT = 1200

# 绘制颜色 (BGR)
COLOR_BBOX = (0, 255, 0)       # 绿色 bbox
COLOR_TRAIL = (0, 200, 255)    # 橙色轨迹线
BBOX_THICKNESS = 2
TRAIL_MAX_LEN = 50             # 每个 ID 最多保留多少帧的轨迹点


def get_center(xyxy):
    """从 xyxy bbox 计算中心点"""
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_bbox_only(frame, xyxy, color=COLOR_BBOX, thickness=BBOX_THICKNESS):
    """只画绿色 bbox，不写任何文字"""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_trail(frame, trail_points, color=COLOR_TRAIL, thickness=2):
    """画轨迹线（polyline）"""
    if len(trail_points) < 2:
        return frame
    pts = np.array(trail_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)
    return frame


def process_video(
    video_path, engine_path, mode="detect", output_dir=OUTPUT_DIR,
    conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, tracker="bytetrack.yaml"
):
    """
    处理视频
    mode: "detect" | "track"
    """
    engine_name = Path(engine_path).stem  # best / best_fp16
    out_dir = Path(output_dir) / f"{engine_name}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    out_video_path = out_dir / f"{video_stem}_{mode}_{engine_name}.mp4"

    print(f"{'='*60}")
    print(f"TensorRT 视频推理")
    print(f"{'='*60}")
    print(f"Engine:       {engine_path}")
    print(f"Mode:         {mode}")
    print(f"Video:        {video_path}")
    print(f"Output size:  {OUT_WIDTH}x{OUT_HEIGHT}")
    print(f"Output:       {out_video_path}")
    if mode == "track":
        print(f"Tracker:      {tracker}")
    print(f"{'='*60}\n")

    # ── 加载模型 ──
    print("加载 TensorRT engine ...")
    model = YOLO(str(engine_path), task="detect")

    # ── 打开视频 ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"源视频: {src_w}x{src_h}, {fps:.1f} FPS, {total_frames} 帧")
    print(f"输出视频: {OUT_WIDTH}x{OUT_HEIGHT}\n")

    # ── 写入器（优先 H.264，回退 mp4v）──
    out_path_str = str(out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path_str, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))
    if not writer.isOpened():
        out_path_str = str(out_video_path.with_suffix(".avi"))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path_str, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))
        out_video_path = Path(out_path_str)
        print(f"  (H264 不可用，回退到 XVID/AVI)")

    # ── 轨迹存储 (track mode) ──
    trails = defaultdict(list)  # {track_id: [(cx, cy), ...]}

    # ── Warmup ──
    print(f"Warmup {WARMUP_FRAMES} 帧 ...")
    warmup_frames = []
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))
        warmup_frames.append(resized)

    for frame in warmup_frames:
        if mode == "track":
            _ = model.track(source=frame, conf=conf, iou=iou, imgsz=IMG_SIZE,
                            device=DEVICE, tracker=tracker, verbose=False)
        else:
            _ = model.predict(source=frame, conf=conf, iou=iou, imgsz=IMG_SIZE,
                              device=DEVICE, verbose=False)
    print("Warmup 完成\n")

    # ── 重新开始读取 ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── 正式推理 ──
    print("开始推理 ...")
    frame_count = 0
    total_detections = 0
    infer_times = []
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 统一 resize 到 1920x1200
        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

        frame_count += 1
        t0 = time.perf_counter()

        if mode == "track":
            result = model.track(
                source=frame, conf=conf, iou=iou, imgsz=IMG_SIZE,
                device=DEVICE, tracker=tracker, persist=True, verbose=False
            )[0]
        else:
            result = model.predict(
                source=frame, conf=conf, iou=iou, imgsz=IMG_SIZE,
                device=DEVICE, verbose=False
            )[0]

        t1 = time.perf_counter()
        infer_times.append(t1 - t0)

        # ── 绘制（bbox 坐标已经是 1920x1200 空间的）──
        out_frame = frame.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            total_detections += len(result.boxes)

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                draw_bbox_only(out_frame, xyxy)

                # 跟踪模式：画轨迹
                if mode == "track" and hasattr(box, 'id') and box.id is not None:
                    tid = int(box.id[0])
                    unique_ids.add(tid)
                    cx, cy = get_center(xyxy)
                    trails[tid].append((cx, cy))
                    if len(trails[tid]) > TRAIL_MAX_LEN:
                        trails[tid] = trails[tid][-TRAIL_MAX_LEN:]
                    draw_trail(out_frame, trails[tid])

        writer.write(out_frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            det_count = len(result.boxes) if result.boxes is not None else 0
            print(f"  [{frame_count}/{total_frames}] 检测: {det_count}")

    writer.release()
    cap.release()

    # ── 统计 ──
    times_arr = np.array(infer_times)

    print(f"\n{'='*60}")
    print(f"推理统计 — {engine_name} ({mode})")
    print(f"{'='*60}")
    print(f"总帧数:            {frame_count}")
    print(f"总检测数:          {total_detections}")
    if mode == "track":
        print(f"唯一跟踪 ID:      {len(unique_ids)}")
    print("-" * 60)
    print(f"平均每帧推理时间:  {times_arr.mean() * 1000:.2f} ms")
    print(f"最快:              {times_arr.min() * 1000:.2f} ms")
    print(f"最慢:              {times_arr.max() * 1000:.2f} ms")
    print(f"中位数:            {np.median(times_arr) * 1000:.2f} ms")
    print(f"FPS (推理):        {1.0 / times_arr.mean():.1f}")
    print(f"FPS (视频原始):    {fps:.1f}")
    print(f"输出视频:          {out_video_path.resolve()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="TensorRT 视频推理：检测 + 跟踪")
    parser.add_argument("--engine", type=str, required=True,
                        help="TensorRT engine 文件路径")
    parser.add_argument("--video", type=str, default=VIDEO_PATH,
                        help=f"输入视频路径（默认: {VIDEO_PATH}）")
    parser.add_argument("--mode", type=str, default="detect",
                        choices=["detect", "track", "both"],
                        help="模式: detect（检测）/ track（跟踪）/ both（都跑）")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help=f"输出目录（默认: {OUTPUT_DIR}）")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"置信度阈值（默认: {CONF_THRESHOLD}）")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                        help=f"IoU 阈值（默认: {IOU_THRESHOLD}）")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="跟踪器（默认: bytetrack.yaml，可选 botsort.yaml）")
    args = parser.parse_args()

    # 解析 engine 路径 (支持简写如 best_n.engine)
    engine_path = Path(args.engine)
    if not engine_path.exists():
        # 尝试在各 train 目录下查找
        for subdir in Path("runs/detect").glob("train_*/weights"):
            alt = subdir / engine_path.name
            if alt.exists():
                engine_path = alt
                break
        # 兼容旧路径
        if not engine_path.exists():
            alt = Path("runs/detect/train/weights") / engine_path.name
            if alt.exists():
                engine_path = alt
            else:
                print(f"错误: 找不到 engine 文件 {args.engine}")
                return

    modes = ["detect", "track"] if args.mode == "both" else [args.mode]
    for mode in modes:
        process_video(
            video_path=args.video,
            engine_path=str(engine_path),
            mode=mode,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
        )
        print()


if __name__ == "__main__":
    main()
