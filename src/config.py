from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
import os

class Config:
    def __init__(self):
        self.model_filename = "yolov8s.pt"
        self.video_path = "data/videos/2.mp4"
        self.colormap = cv2.COLORMAP_CIVIDIS
        self.decay_factor = 0.99
        self.output_report_path = "C:/Users/pnrde/Desktop/super_heatmapper_project/results/log_files/report.txt"
        self.output_heatmap_folder = "C:/Users/pnrde/Desktop/super_heatmapper_project/results/generated_heatmaps/"

def main():
    config = Config()

    # Model dosyasının tam yolu
    model_path = os.path.join("models", config.model_filename)
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(config.video_path)
    assert cap.isOpened(), "Error reading video file"

    # Heatmap Init
    heatmap_obj = heatmap.Heatmap()
    heatmap_obj.set_args(colormap=config.colormap,
                         imw=cap.get(4),
                         imh=cap.get(3),
                         view_img=True,
                         decay_factor=config.decay_factor)

    report_lines = []  # Rapor satırlarını depolamak için bir liste
    frame_count = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.track(im0, persist=True)
        heatmap_image = heatmap_obj.generate_heatmap(im0, tracks=results)

        # Burada rapor satırları oluşturulur (istediğiniz bilgileri ekleyebilirsiniz)
        frame_count += 1
        report_line = f"Frame: {frame_count}, Objects: {len(results)}\n"
        report_lines.append(report_line)

        # Heatmap'i kaydet
        heatmap_filename = f"heatmap_frame_{frame_count}.png"
        heatmap_path = os.path.join(config.output_heatmap_folder, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_image)

    # Raporu dosyaya yazma
    with open(config.output_report_path, "w") as report_file:
        report_file.writelines(report_lines)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
