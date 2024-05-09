import os
import cv2
from pathlib import Path
from app.yolov9 import YOLOv9
from ultralytics import YOLO
from timer import Timer
    
def yolo_run_image(args):
    """
    Run YOLO model on a single image and return bounding box detections.

    Args:
        args: Arguments containing image path, model weights, device type, and display options.

    Returns:
        list: 2D array with detected class, confidence, and bounding box coordinates.
        float: Elapsed time for inference.
    """
    timer = Timer()
    conf_bb = []

    print("[INFO] Initialize Model")
    model = YOLO(args.weights, task="detect")
    # model.to(args.device)

    source_path = args.source
    assert os.path.isfile(source_path), f"Source file {source_path} does not exist."
    image = cv2.imread(source_path)

    print("[INFO] Inference Image")
    timer.start()
    # results = model(image)
    results = model.predict(source=image, device=args.device)
    
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

    timer.stop()
    elapsed_time = timer.result()
    print(f"Elapsed time: {elapsed_time:0.4f} seconds")

    if args.show:
        cv2.imshow("Result", results[0].imgs[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return conf_bb, elapsed_time

def yolo_run_video(args):
    """
    Runs YOLO model on a video stream.

    Args:
        args: Arguments containing video path.

    Returns:
        float: Elapsed time for inference.
    """
    timer = Timer()

    print("[INFO] Initialize Model")
    model = YOLO(args.weights)

    print("[INFO] Inference on Video")
    timer.start()
    cap = cv2.VideoCapture(args.source)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

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

def get_detector(args):
    """
    Get detector (YOLOv9 Model) for ONNX runtime

    Args:
        args: Arguments containing data path.

    Returns:
        object: YOLOv9
    """
    weights_path = args.weights
    classes_path = args.classes
    source_path = args.source
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

    if args.type == "image":
        image = cv2.imread(source_path)
        h,w = image.shape[:2]
    elif args.type == "video":
        cap = cv2.VideoCapture(source_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = YOLOv9(model_path=weights_path,
                      class_mapping_path=classes_path,
                      original_size=(w, h),
                      score_threshold=args.score_threshold,
                      conf_thresold=args.conf_threshold,
                      iou_threshold=args.iou_threshold,
                      device=args.device)
    return detector

def inference_on_image(args):
    """
    Run ONNX model inference on a single image and return detected bounding boxes.

    Args:
        args: Contains path to source image, model configuration, and display options.

    Returns:
        list: 2D array with detected class, confidence, and bounding box coordinates.
        float: Elapsed time for inference.
    """
    timer = Timer()
    conf_bb = []
    
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    image = cv2.imread(args.source)
    
    timer.start()
    print("[INFO] Inference Image")
    detections = detector.detect(image)
    
    for detection in detections:
        box = detection["box"]
        x1, y1, x2, y2 = box
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        conf_bb.append([class_name, float(confidence), x1, y1, x2, y2])
        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

    # detector.draw_detections(image, detections=detections)
    timer.stop()
    elapsed_time = timer.result()
    print(f"Elapsed time: {elapsed_time:0.4f} seconds")

    if args.show:
        output_path = f"./app/output/{Path(args.source).name}"
        cv2.imwrite(output_path, image)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
    
    return conf_bb, elapsed_time

def inference_on_video(args):
    """
    Runs YOLO model on a video stream.

    Args:
        args: Arguments containing video path and display options.

    Returns:
        float: Elapsed time for inference.
    """
    timer = Timer()
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    
    print("[INFO] Inference on Video")
    cap = cv2.VideoCapture(args.source)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # writer = cv2.VideoWriter('./app/output/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))

    timer.start()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        # detector.draw_detections(frame, detections=detections)
        # writer.write(frame)
        
        if args.show:
            cv2.imshow("Result", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
    timer.stop()
    elapsed_time = timer.result()
    print(f"Elapsed time: {elapsed_time:0.4f} seconds")
    
    return elapsed_time