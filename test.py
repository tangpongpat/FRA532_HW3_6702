import cv2
import torch
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self,
                 model_path: str,
                 input_path: str,
                 output_path: str,
                 skip_frames: int = 3,
                 target_width: int = 640):
        """
        Initialize the video processor.
        Args:
            model_path: path or name of the YOLO model
            input_path: path to input video file
            output_path: where to save processed video
            skip_frames: process every nth frame
            target_width: resize width for inference
        """
        self.cap = cv2.VideoCapture(input_path)
        assert self.cap.isOpened(), "Cannot open input video"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) *
                (target_width / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, h))
        self.skip = skip_frames
        self.width = target_width

        # Load model and push to GPU if available
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def _preprocess(self, frame):
        """Resize and convert BGR to RGB."""
        frame_resized = cv2.resize(frame,
                                   (self.width,
                                    int(frame.shape[0] * self.width / frame.shape[1])))
        return cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    def _draw_boxes(self, frame, results):
        """Draw detection boxes and labels back onto BGR frame."""
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame

    def process(self, batch_size: int = 4, conf: float = 0.5):
        """
        Process the entire video.
        Args:
            batch_size: number of frames per batch inference
            conf: confidence threshold
        """
        frame_buffer = []
        try:
            idx = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if idx % self.skip == 0:
                    pre = self._preprocess(frame)
                    frame_buffer.append((frame, pre))

                # Once buffer is full, infer and write
                if len(frame_buffer) >= batch_size:
                    originals, inputs = zip(*frame_buffer)
                    results = self.model(inputs,
                                         conf=conf,
                                         device=self.device,
                                         batch_size=batch_size)
                    for orig, res in zip(originals, results):
                        out = self._draw_boxes(orig, [res])
                        self.writer.write(out)
                    frame_buffer.clear()
                idx += 1

            # Process any remaining frames
            if frame_buffer:
                originals, inputs = zip(*frame_buffer)
                results = self.model(inputs,
                                     conf=conf,
                                     device=self.device,
                                     batch_size=len(inputs))
                for orig, res in zip(originals, results):
                    out = self._draw_boxes(orig, [res])
                    self.writer.write(out)

        except KeyboardInterrupt:
            print("Interrupted by user; finalizing video…")
        finally:
            self.cap.release()
            self.writer.release()
            print("Processing complete. Output saved to:", self.writer)

if __name__ == "__main__":
    vp = VideoProcessor(
        model_path="yolov8n.pt",
        input_path="input.mp4",
        output_path="output.mp4",
        skip_frames=5,     # process every 5th frame
        target_width=640   # resize width to 640 px
    )
    vp.process(batch_size=8, conf=0.5)
