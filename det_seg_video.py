"""
det_seg_video.py
================
检测(ByteTrack) + 分割(ENet) 联合推理脚本

流程:
1. 输入视频必须是 1920x1200（先用 prepare_video.py 或手动 resize）
2. TensorRT 检测，得到每个精子头部的 bbox
3. 以 bbox 中心裁剪 64x64 ROI，送入 ENet 分割模型
4. 将 64x64 分割 mask 叠加回原帧对应位置
5. ByteTrack 跟踪，画轨迹
6. 输出 4 种视频:
   - full:       检测框 + 轨迹 + 分割叠加
   - det_track:  检测框 + 轨迹
   - seg:        仅分割叠加
   - det_seg:    检测框 + 分割叠加

用法:
    conda run -n yolo python det_seg_video.py --engine best.engine --video gray_video_1920x1200.mp4
"""

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"

import sys
import cv2
import torch
import numpy as np
import argparse
import yaml
import time
from collections import defaultdict

# ── 添加分割项目路径 ──
SEG_ROOT = "/root/Segmentation-Model-Zoo-for-Super-Small-Object"
sys.path.insert(0, SEG_ROOT)

# ── 常量 ──
ROI_SIZE = 64
DEVICE = "cuda:1"

# 颜色 (BGR)
COLOR_BBOX  = (0, 255, 0)      # 绿色 - 检测框
COLOR_TRAIL = (0, 200, 255)    # 橙色 - 轨迹
# 分割颜色: class_index -> BGR
SEG_COLORS = {
    0: (0, 0, 255),            # 红色 - non-measurable head
    1: (0, 255, 0),            # 绿色 - nuclear
    2: (255, 0, 0),            # 蓝色 - acrosome
    3: (0, 255, 255),          # 黄色 - mid-piece
}

# 轨迹
track_history = defaultdict(list)
MAX_TRAIL_LEN = 60


# ============================================================
#  模型加载
# ============================================================

def load_seg_model(seg_model_dir, device):
    """加载 ENet 分割模型"""
    from model_zoo.enet import ENet

    config_path = os.path.join(seg_model_dir, "config.yml")
    weight_path = os.path.join(seg_model_dir, "model.pth")

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = ENet(
        config['num_classes'],
        config['input_channels'],
        config.get('deep_supervision', False)
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config


# ============================================================
#  ROI 裁剪 & 分割推理
# ============================================================

def crop_roi_centered(frame, cx, cy, size=ROI_SIZE):
    """
    以 (cx, cy) 为中心裁剪 size x size 的 ROI。
    越界部分用黑色填充。
    返回: roi [size, size, 3], rx1, ry1 (ROI 左上角在 frame 上的坐标)
    """
    h, w = frame.shape[:2]
    half = size // 2
    rx1, ry1 = cx - half, cy - half
    rx2, ry2 = rx1 + size, ry1 + size

    # frame 上的有效区域
    sx1, sy1 = max(0, rx1), max(0, ry1)
    sx2, sy2 = min(w, rx2), min(h, ry2)
    # ROI 上的偏移
    dx1, dy1 = sx1 - rx1, sy1 - ry1
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)

    roi = np.zeros((size, size, 3), dtype=np.uint8)
    roi[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    return roi, rx1, ry1


def seg_inference(model, roi_bgr, device):
    """
    对 64x64 BGR ROI 做分割推理。
    预处理: 保持 BGR, /255, ImageNet归一化, 再 /255 (复现训练时 dataset.py 的双重归一化)
    返回: masks [4, 64, 64] uint8 (0 or 255)
    """
    # 保持 BGR (训练时 cv2.imread 默认 BGR，不做 RGB 转换)
    roi_float = roi_bgr.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    roi_norm = (roi_float - mean) / std
    # 训练代码 dataset.py 在 A.Normalize() 后又 /255，这里复现同样的行为
    roi_final = roi_norm / 255.0
    # HWC -> CHW -> tensor
    tensor = torch.from_numpy(roi_final.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, list):
            output = output[-1]
        prob = torch.sigmoid(output).cpu().numpy()[0]  # [4, 64, 64]

    masks = (prob >= 0.5).astype(np.uint8) * 255       # [4, 64, 64]
    return masks


# ============================================================
#  分割叠加
# ============================================================

def overlay_seg(frame, masks, rx1, ry1, alpha=0.4):
    """
    将 64x64 的分割 masks 叠加到 frame 上对应位置。
    masks: [4, 64, 64] uint8 (0 or 255)
    rx1, ry1: 64x64 ROI 在 frame 上的左上角坐标
    """
    h, w = frame.shape[:2]
    size = masks.shape[1]  # 64

    # ROI 在 frame 上的有效范围
    fx1, fy1 = max(0, rx1), max(0, ry1)
    fx2, fy2 = min(w, rx1 + size), min(h, ry1 + size)
    if fx1 >= fx2 or fy1 >= fy2:
        return frame

    # mask 上对应的范围
    mx1, my1 = fx1 - rx1, fy1 - ry1
    mx2, my2 = mx1 + (fx2 - fx1), my1 + (fy2 - fy1)

    roi_region = frame[fy1:fy2, fx1:fx2].astype(np.float32)

    for cls_idx in range(4):
        mask_crop = masks[cls_idx][my1:my2, mx1:mx2]
        if mask_crop.max() == 0:
            continue
        color = SEG_COLORS[cls_idx]
        binary = (mask_crop > 0).astype(np.float32)[..., np.newaxis]  # (H,W,1)
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        roi_region = roi_region * (1 - binary * alpha) + color_arr * binary * alpha

    frame[fy1:fy2, fx1:fx2] = roi_region.clip(0, 255).astype(np.uint8)
    return frame


# ============================================================
#  绘制
# ============================================================

def draw_bbox(frame, cx, cy, size=ROI_SIZE, color=COLOR_BBOX, thickness=2):
    """画正方形检测框"""
    half = size // 2
    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), color, thickness)


def draw_trails(frame, history, color=COLOR_TRAIL, thickness=2):
    """画所有 track 的轨迹"""
    for _, points in history.items():
        if len(points) < 2:
            continue
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, thickness)


# ============================================================
#  VideoWriter
# ============================================================

def create_writer(output_path, fps, width, height):
    """创建 VideoWriter，优先 mp4v，fallback XVID/AVI"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, output_path

    print("  [!] mp4v 不可用，使用 XVID/AVI")
    fallback = output_path.rsplit('.', 1)[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(fallback, fourcc, fps, (width, height))
    return writer, fallback


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="检测+分割 联合推理")
    parser.add_argument("--engine", required=True, help="TensorRT engine 路径")
    parser.add_argument("--video", required=True, help="输入视频 (必须是 1920x1200)")
    parser.add_argument("--seg_model", default=os.path.join(SEG_ROOT, "models/ENet_sperm_ROINAHead_Xiangya_260817"),
                        help="分割模型目录")
    parser.add_argument("--output", default="det_seg_results", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器")
    parser.add_argument("--seg_alpha", type=float, default=0.4, help="分割叠加透明度")
    parser.add_argument("--device", default=DEVICE, help="设备")
    args = parser.parse_args()

    # ── 加载模型 ──
    print(f"[1/3] 加载检测 engine: {args.engine}")
    from ultralytics import YOLO
    det_model = YOLO(args.engine)

    print(f"[2/3] 加载分割模型: {args.seg_model}")
    seg_model, seg_config = load_seg_model(args.seg_model, args.device)
    print(f"  arch={seg_config['arch']}, classes={seg_config['num_classes']}, "
          f"input={seg_config['input_h']}x{seg_config['input_w']}")

    # ── 打开视频 ──
    print(f"[3/3] 打开视频: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  输入: {src_w}x{src_h}, {fps:.1f} FPS, {total_frames} 帧")

    if src_w != 1920 or src_h != 1200:
        print(f"  [WARNING] 视频不是 1920x1200，当前为 {src_w}x{src_h}，结果可能不正确")

    # ── 输出 ──
    os.makedirs(args.output, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    engine_name = os.path.splitext(os.path.basename(args.engine))[0]
    prefix = f"{video_name}_{engine_name}"

    writers = {}
    output_paths = {}
    for key in ['full', 'det_track', 'seg', 'det_seg']:
        w, p = create_writer(os.path.join(args.output, f"{prefix}_{key}.mp4"), fps, src_w, src_h)
        writers[key] = w
        output_paths[key] = os.path.basename(p)

    # ── 逐帧处理 ──
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测 + 跟踪
        results = det_model.track(
            frame, persist=True, tracker=args.tracker,
            conf=args.conf, iou=args.iou,
            verbose=False, device=args.device,
        )
        result = results[0]

        # 4 个输出副本
        out_frames = {
            'full':      frame.copy(),
            'det_track': frame.copy(),
            'seg':       frame.copy(),
            'det_seg':   frame.copy(),
        }

        if result.boxes is not None and result.boxes.id is not None:
            xyxys = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for i in range(len(xyxys)):
                x1, y1, x2, y2 = xyxys[i].astype(int)
                track_id = ids[i]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 更新轨迹
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > MAX_TRAIL_LEN:
                    track_history[track_id] = track_history[track_id][-MAX_TRAIL_LEN:]

                # 以中心裁剪 64x64 ROI → 分割
                roi, rx1, ry1 = crop_roi_centered(frame, cx, cy)
                masks = seg_inference(seg_model, roi, args.device)

                # 画检测框
                for key in ['full', 'det_track', 'det_seg']:
                    draw_bbox(out_frames[key], cx, cy)

                # 叠加分割 mask
                for key in ['full', 'seg', 'det_seg']:
                    overlay_seg(out_frames[key], masks, rx1, ry1, alpha=args.seg_alpha)

        # 画轨迹
        for key in ['full', 'det_track']:
            draw_trails(out_frames[key], track_history)

        # 写入
        for key, writer in writers.items():
            writer.write(out_frames[key])

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  帧 {frame_idx}/{total_frames} | {elapsed / frame_idx * 1000:.1f} ms/帧")

    # ── 释放 ──
    cap.release()
    for writer in writers.values():
        writer.release()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"完成: {frame_idx} 帧, {elapsed:.1f}s ({elapsed / frame_idx * 1000:.1f} ms/帧)")
    print(f"输出: {args.output}/")
    for key in ['full', 'det_track', 'seg', 'det_seg']:
        print(f"  {key}: {output_paths[key]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
