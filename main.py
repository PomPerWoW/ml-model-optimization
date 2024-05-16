import os
import cv2
from pathlib import Path
from app.yolov9_ultralytics.models.yolo import YOLOv9
from app.yolov9_onnxruntime import YoloV9Onnxruntime
from app.yolov9_openvino import YoloV9Openvino
from app.util.timer import Timer
import pdb

class YoloRuntimeTest:
    def __init__(self):
        pass

    @staticmethod
    def _initialize_ultralytics_model(args):
        """Initialize the YOLOv9 model with the given weights and task."""
        print("[INFO] Initialize Model")
        model = YOLOv9(args["weights"], task="detect")
        return model

    @staticmethod
    def _display_result_window(image, show):
        """Display the image in a window if the 'show' flag is True."""
        if show:
            cv2.imshow("Result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def _initialize_onnxruntime_model(args):
        """Initialize the ONNX Runtime detector with the given arguments."""
        print("[INFO] Initialize Model")
        weights_path = args["weights"]
        classes_path = args["classes"]
        source_path = args["source"]  
        assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
        assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
        assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

        if args["type"] == "image":
            image = cv2.imread(source_path)
            h, w = image.shape[:2]
        elif args["type"] == "video":
            cap = cv2.VideoCapture(source_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector = YoloV9Onnxruntime(model_path=weights_path,
                                     class_mapping_path=classes_path,
                                     original_size=(w, h),
                                     conf_threshold=args["conf_threshold"],
                                     iou_threshold=args["iou_threshold"],
                                     device=args["device"])
        return detector

    @staticmethod
    def _inference_on_image(detector, args):
        """Perform inference on a single image and return the detected objects and elapsed time."""
        timer = Timer()
        conf_bb = []
        elapsed_time = 0

        print("[INFO] Inference Image")
        timer.start()
        
        if hasattr(detector, 'detect'):
            image = cv2.imread(args["source"])
            detections = detector.detect(image)
        else:
            detections = detector.predict(source=args["source"], device=args["device"], imgsz=640, conf=args["conf_threshold"], iou=args["iou_threshold"])

        timer.stop()
        elapsed_time = timer.result()
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
        if hasattr(detector, 'detect'):
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
                print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
        else:
            for detection in detections:
                for box in detection.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = detector.names[class_id]
                    conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
                    print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

        if args["show"]:
            output_path = f"./app/output/image_output.jpg"
            cv2.imwrite(output_path, args["source"])
            YoloRuntimeTest._display_result_window(args["source"], args["show"])

        return conf_bb, elapsed_time

    @staticmethod
    def _inference_on_video(detector, args):
        """Perform inference on a video file and return the elapsed time."""
        timer = Timer()

        print("[INFO] Inference on Video")
        timer.start()
        cap = cv2.VideoCapture(args["source"])

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                detections = detector.detect(frame)

                if args.show:
                    cv2.imshow("Result", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        timer.stop()
        elapsed_time = timer.result()
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

        return elapsed_time

    def ultralytics_run_image(self, args):
        """Run inference on an image using the Ultralytics YOLO model."""
        model = self._initialize_ultralytics_model(args)
        return self._inference_on_image(model, args)

    def ultralytics_run_video(self, args):
        """Run inference on a video using the Ultralytics YOLO model."""
        model = self._initialize_ultralytics_model(args)
        return self._inference_on_video(model, args)

    def onnxruntime_run_image(self, args):
        """Run inference on an image using the ONNX Runtime model."""
        detector = self._initialize_onnxruntime_model(args)
        return self._inference_on_image(detector, args)

    def onnxruntime_run_video(self, args):
        """Run inference on a video using the ONNX Runtime model."""
        detector = self._initialize_onnxruntime_model(args)
        return self._inference_on_video(detector, args)

    @staticmethod
    def export_onnx():
        """Export the YOLOv9 model to ONNX format."""
        model = YOLOv9('./app/weights/yolov9c.pt')
        model.export(format='onnx')

    @staticmethod
    def export_openvino():
        """Export the YOLOv9 model to OpenVINO format."""
        model = YOLOv9('./app/weights/yolov9c.pt')
        model.export(format='openvino')

class LightGlueRuntimeTest(YoloRuntimeTest):
    def __init__(self):
        """Initialize the LightGlueRuntimeTest class, inheriting from YoloRuntimeTest."""
        super().__init__()
