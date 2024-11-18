from ultralytics import YOLO
import cv2
import numpy as np

class YOLO_detector:
    def __init__(self, model_path:str):
        '''
        Initialize the YOLOv8 object with the specified model.
        :param model_path: Path to the YOLOv8 model file.
        '''
        
        self.model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        
    def infer_image(self, image_path: str, conf_threshold: float = 0.5, output_path: str = None):   
        """
        Perform inference on an image using the YOLOv8 model.
        :param image_path: Path to the input image.
        :param conf_threshold: Confidence threshold for detections.
        :return: List of detections.
        """
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        results = self.model.predict(source=image, conf=conf_threshold)
        
        if output_path is None: 
            output_path = image_path.replace(".png", "_output.png")
            
        annotated_image = results[0].plot()
        cv2.imwrite(output_path, annotated_image)
        print(f"Image predict completed. Results saved to {output_path}")
    
    
    
    def infer_video(self, video_path: str, conf_threshold: float = 0.5, output_path: str = None):
        """
        Perform inference on a video using the YOLOv8 model.
        :param video_path: Path to the input video.
        :param conf_threshold: Confidence threshold for detections.
        :return: List of detections.
        """
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found at {video_path}")
        
        if output_path is None: 
            output_path = video_path.replace(".mp4", "_output.mp4")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(source=frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"Video predict completed. Results saved to {output_path}")


    def infer_webcam(self, webcam_index: int = 0, conf_threshold: float = 0.5):
        '''
        Performs inference on a webcam stream in realtime

        :webcam_index: index of the webcam (Default = 0)
        :param conf_threshold: Confidence threshold for detections.
        '''

        cap = cv2.VideoCapture(webcam_index)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.predict(source=frame, conf=conf_threshold)
            annotated_frame = results[0].plot()

            cv2.imshow('Webcam inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam inference...")
                break

        cap.release()
        cv2.destroyAllWindows()