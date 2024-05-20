import pyrootutils
from app.util.timer import Timer

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List

class YOLOv9Onnxruntime:
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 original_size: Tuple[int, int] = (1280, 720),
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "CPU") -> None:
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path

        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size
        self.timer = Timer()
        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() != "cpu":
            providers = ['CUDAExecutionProvider']
        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, 'r') as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        for i in range(image_data.shape[0]):
            print(image_data[i])
        print(image_data.shape)
        
        print(image_data[0,0,0,0:10])
        return image_data
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 
    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        detections = []
        for bbox, score, label in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": self.get_label_name(label)
            })
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]
        
    def detect(self, img: np.ndarray) -> List:
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        print(f'Hello this is output na: {outputs}')
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: List):
        for detection in detections:
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']
            color = self.color_palette[class_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)