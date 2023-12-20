import cv2
from ultralytics import YOLO

def initialize_yolo(model_path="yolov8s.pt"):
    """
    YOLO modelini başlatır ve döndürür.
    """
    model = YOLO(model_path)
    return model

def process_image_with_yolo(model, image_path):
    """
    Verilen bir resmi YOLO modeli kullanarak işler.
    """
    image = cv2.imread(image_path)
    results = model(image)
    return results

def draw_boxes_on_image(image, results):
    """
    YOLO tarafından tespit edilen nesnelerin kutularını resme çizer.
    """
    for box in results.xyxy[0].numpy():
        x_min, y_min, x_max, y_max, conf, cls = box
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        label = f"Class {int(cls)}: {conf:.2f}"
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        image = cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
