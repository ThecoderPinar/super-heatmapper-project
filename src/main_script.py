from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model
cap = cv2.VideoCapture("data/videos/2.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               int(cap.get(5)),
                               (int(cap.get(3)), int(cap.get(4))))

# Heatmap Init
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                     imw=cap.get(4),  # should same as cap width
                     imh=cap.get(3),  # should same as cap height
                     view_img=True,
                     decay_factor=0.99)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
      print("Video frame is empty or video processing has been successfully completed.")
      break

    results = model.track(im0, persist=True)
    im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
    video_writer.write(im0)
    
video_writer.release()
cv2.destroyAllWindows()