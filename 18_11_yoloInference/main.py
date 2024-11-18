from src.yolo_detector import YOLO_detector

if __name__ == "__main__":
    # Initialize YOLOv8 object
    yolo = YOLO_detector("weights/yolov8s.pt")

    # Perform prediction image
    yolo.infer_image("assets/images/img.png", conf_threshold=0.5)

    # # Perform prediction on video
    # yolo.infer_video("assets/videos/test.mov", conf_threshold=0.5, output_path="assets/videos/test_output.mp4")
    
    # Perform prediction on webcam
    yolo.infer_webcam(0, conf_threshold=0.5)