import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

def predict_video_with_tensorrt(
    video_path,
    engine_path,
    output_dir="output",
    conf_threshold=0.25,
    iou_threshold=0.7,
    save_txt=True
):
    """
    使用 TensorRT 模型预测视频并保存结果
    
    Args:
        video_path: 输入视频路径
        engine_path: TensorRT engine 文件路径
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IOU 阈值
        save_txt: 是否保存预测日志到txt
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_video = output_path / f"{video_name}_result_{timestamp}.mp4"
    log_file = output_path / f"{video_name}_log_{timestamp}.txt"
    
    # 重定向 stdout 到日志文件
    if save_txt:
        log_handler = open(log_file, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = log_handler
    
    try:
        print(f"=" * 80)
        print(f"视频预测开始")
        print(f"=" * 80)
        print(f"输入视频: {video_path}")
        print(f"TensorRT 模型: {engine_path}")
        print(f"输出视频: {output_video}")
        print(f"置信度阈值: {conf_threshold}")
        print(f"IOU 阈值: {iou_threshold}")
        print(f"=" * 80)
        print()
        
        # 加载 TensorRT 模型
        print("正在加载 TensorRT 模型...")
        model = YOLO(engine_path, task="detect")
        print("模型加载成功！")
        print()
        
        # 使用 ultralytics 官方的 predict 方法处理视频
        # save=True 会自动保存可视化后的视频
        print("开始预测视频...")
        results = model.predict(
            source=video_path,
            save=True,  # 保存可视化结果
            conf=conf_threshold,  # 置信度阈值
            iou=iou_threshold,  # NMS IOU 阈值
            show=False,  # 不显示实时预测窗口
            stream=True,  # 使用流式处理（节省内存）
            verbose=True,  # 显示详细信息
            project=str(output_path),  # 保存到指定目录
            name=f"{video_name}_predict_{timestamp}",  # 子目录名称
            exist_ok=True  # 如果目录存在则覆盖
        )
        
        # 处理每一帧的结果
        frame_count = 0
        total_detections = 0
        
        for result in results:
            frame_count += 1
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            # 每10帧打印一次信息
            if frame_count % 10 == 0:
                print(f"帧 {frame_count}: 检测到 {num_detections} 个目标")
            
            # 打印每个检测框的详细信息
            if num_detections > 0:
                for idx, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    class_name = model.names[cls]
                    
                    if frame_count % 30 == 0:  # 每30帧打印一次详细信息
                        print(f"  检测 {idx+1}: 类别={class_name}, 置信度={conf:.3f}, "
                              f"坐标=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
        print()
        print(f"=" * 80)
        print(f"预测完成！")
        print(f"=" * 80)
        print(f"总帧数: {frame_count}")
        print(f"总检测数: {total_detections}")
        print(f"平均每帧检测数: {total_detections/frame_count if frame_count > 0 else 0:.2f}")
        print(f"结果视频保存在: {output_path / f'{video_name}_predict_{timestamp}'}")
        print(f"=" * 80)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 恢复 stdout
        if save_txt:
            sys.stdout = original_stdout
            log_handler.close()
            print(f"Predicted log has been saved to: {log_file}")


if __name__ == "__main__":
    # 配置参数
    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/Basler acA1920-155ucMED (40214438)_20241105_115335174.mp4"  
    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/Basler acA1920-155ucMED (40214438)_20241105_115426118.mp4"  
    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/Basler acA1920-155ucMED (40214438)_20241105_115540765.mp4"  

    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/using1.mp4"  
    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/using2.mp4" 
    # VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/using3.mp4"
    VIDEO_PATH = r"/root/yolo26-tensorrt/dataset/test_videoes/using4.mp4"  



    ENGINE_PATH = r"/root/yolo26-tensorrt/runs/detect/train/weights/best.engine"
    OUTPUT_DIR = r"video_results"
    
    # 运行预测
    predict_video_with_tensorrt(
        video_path=VIDEO_PATH,
        engine_path=ENGINE_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25,  # 可调整置信度阈值
        iou_threshold=0.7     # 可调整 IOU 阈值
    )