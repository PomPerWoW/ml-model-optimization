import os
import cv2
import torch
import yaml
from pathlib import Path
from app.yolov9_ultralytics.models.yolo import YOLOv9
from app.yolov9_onnxruntime import YOLOv9Onnxruntime
from app.yolov9_openvino import YOLOv9Openvino
from app.util.timer import Timer
from app.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from app.LightGlue.lightglue.utils import load_image, rbd
from app.LightGlue.lightglue import viz2d
from app.LightGlueOnnx.export import export_onnx
from app.LightGlueOnnx.infer import infer
from threading import Thread

class YoloRuntimeTest:
    def __init__(self):
        pass

    @staticmethod
    def _initialize_ultralytics_model(args):
        """Initialize the YOLOv9 model with the given weights and task."""
        # print("[INFO] Initialize Model")
        weights_path = args["weights"]
        try:
            os.path.isfile(weights_path)
            os.path.isdir(weights_path)
        except:
            raise FileNotFoundError(f"There's no weight file/dir with name {weights_path}")
        
        model = YOLOv9(weights_path, task="detect")
        return model

    @staticmethod
    def _initialize_onnxruntime_model(args):
        """Initialize the ONNX Runtime model with the given arguments."""
        # print("[INFO] Initialize Model")
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
        model = YOLOv9Onnxruntime(model_path=weights_path,
                                    class_mapping_path=classes_path,
                                    original_size=(w, h),
                                    conf_threshold=args["conf_threshold"],
                                    iou_threshold=args["iou_threshold"],
                                    device=args["device"])
        return model
    
    @staticmethod
    def _initialize_openvino_model(args):
        """Initialize the YOLOv9 model with the given weights and task."""
        # print("[INFO] Initialize Model")
        weights_path = args["weights"]
        classes_path = args["classes"]
        assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
        assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
        
        model = YOLOv9Openvino(xml_model_path=weights_path, 
                               classes=classes_path, 
                               conf=args["conf_threshold"], 
                               nms=args["iou_threshold"])
        return model

    @staticmethod
    def _display_result_window(image, show):
        """Display the image in a window if the 'show' flag is True."""
        if show:
            cv2.imshow("Result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def _inference_on_image(model, args):
        """Perform inference on a single image and return the detected objects and elapsed time."""
        timer = Timer()
        conf_bb = []
        elapsed_time = 0
        image = cv2.imread(args["source"])
        inference_type = args["inference_type"]
        
        with open(args["classes"], 'r') as file:
            yaml_file = yaml.safe_load(file)
            classes = yaml_file['names']
            keys_list = sorted(classes.keys())
            
        print("[INFO] Inference Image")
        timer.start()
        
        if inference_type == "ultralytics":
            detections = model.predict(source=image, device=args["device"], imgsz=640, conf=args["conf_threshold"], iou=args["iou_threshold"], classes=keys_list)
        if inference_type == "onnxruntime_model":
            detections = model.detect(image)
        if inference_type == "openvino_model":
            img_resized, dw, dh = model.resize_and_pad(image)
            detections = model.predict(img_resized)

        timer.stop()
        elapsed_time = timer.result()
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

        infer_time = 0.0
        
        if inference_type == "ultralytics":
            for detection in detections:
                infer_time = detection.speed["inference"]
                for box in detection.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
        
        if inference_type == "onnxruntime_model":
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])

        if inference_type == "openvino_model":
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                class_name = detection["class_index"]
                confidence = detection["confidence"]
                conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
        
        if args["show"]:
            output_path = f"./app/output/image_output.jpg"
            cv2.imwrite(output_path, args["source"])
            YoloRuntimeTest._display_result_window(args["source"], args["show"])

        return conf_bb, elapsed_time, infer_time

    @staticmethod
    def _inference_on_video(model, args):
        """Perform inference on a video file and return the elapsed time."""
        timer = Timer()

        print("[INFO] Inference on Video")
        timer.start()
        cap = cv2.VideoCapture(args["source"])

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                detections = model.detect(frame)

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
    
    def openvino_run_image(self, args):
        """Run inference on a video using the Openvino model."""
        model = self._initialize_openvino_model(args)
        return self._inference_on_image(model, args)
    
    def openvino_run_video(self, args):
        """Run inference on a video using the Openvino model."""
        model = self._initialize_openvino_model(args)
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
        torch.set_grad_enabled(False)
        self.images_path = Path("./app/assets/lightglue_assets")
    
    def lightglue_model(self, img1, img2):
        # print("[INFO] Initialize Model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        
        print("[INFO] Inference Model")
        timer = Timer()
        timer.start()
        image0 = load_image(self.images_path / img1)
        image1 = load_image(self.images_path / img2)

        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        timer.stop()
        print(f'time elapsed: {timer.elapsed_time} s')

        # axes = viz2d.plot_images([image0, image1])
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        
        return timer.elapsed_time

    def lightglue_onnx_model(self, img1, img2):
        # print("[INFO] Initialize Model")
        extractor_type = "superpoint"
        extractor_path = f"./app/LightGlueOnnx/weights/{extractor_type}.onnx"
        lightglue_path = f"./app/LightGlueOnnx/weights/{extractor_type}_lightglue.onnx"

        # export_onnx(
        #     extractor_type=extractor_type,
        #     extractor_path=extractor_path,
        #     lightglue_path=lightglue_path,
        #     dynamic=True,
        #     max_num_keypoints=None,
        # )
        
        print("[INFO] Inference Model")
        timer = Timer()
        timer.start()
        m_kpts0, m_kpts1 = infer(
            img_paths=[f'./app/assets/lightglue_assets/{img1}', f'./app/assets/lightglue_assets/{img2}'],
            extractor_type=extractor_type,
            extractor_path=extractor_path,
            lightglue_path=lightglue_path,
            img_size=640,
            viz=False,
        )
        timer.stop()
        print(f'time elapsed: {timer.elapsed_time} s')
        
        return timer.elapsed_time