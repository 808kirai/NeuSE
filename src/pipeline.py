import cv2
import numpy as np
from src.config import config
from src.logger import data_logger


# process dual-spectral frames with reusable clahe and input checks
class dual_spectral_pipeline:
    def __init__(self):
        # set up the database logger
        self.logger = data_logger()
        # reuse clahe to avoid per-frame setup
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process_frames(self, rgb_frame=None, thermal_frame=None):
        try:
            # validate the input frames
            if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            if thermal_frame is None or not isinstance(thermal_frame, np.ndarray) or thermal_frame.size == 0:
                thermal_frame = rgb_frame.copy()

            # convert thermal data to grayscale
            if thermal_frame.ndim == 2:
                thermal_gray = thermal_frame
            elif thermal_frame.ndim == 3 and thermal_frame.shape[2] == 3:
                thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
            else:
                thermal_gray = np.zeros((480, 640), dtype=np.uint8)

            # apply clahe to reduce background clutter
            enhanced_thermal = self.clahe.apply(thermal_gray)

            # isolate high-heat areas
            _, heat_mask = cv2.threshold(enhanced_thermal, 200, 255, cv2.THRESH_BINARY)

            # find heat contours
            contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hits = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # skip noise and oversized artifacts
                if area < 100 or area > 50000:
                    continue

                # calculate the bounding box and confidence
                x, y, w, h = cv2.boundingRect(cnt)
                conf = float(np.clip(0.85 + (area / 10000.0), 0.85, 0.99))

                if conf >= config.conf_threshold:
                    peak_heat = int(np.max(thermal_gray[y:y + h, x:x + w]))
                    box = [x, y, w, h]
                    
                    # save the detection
                    self.logger.log_event(conf, peak_heat, box)
                    hits.append({
                        "bbox": box,
                        "confidence": conf,
                        "thermal_peak": peak_heat,
                    })

            return hits

        except Exception:
            # return no detections for invalid frames
            return []


# preserve compatibility with older callers
class VisionPipeline(dual_spectral_pipeline):
    def process_frame(self, rgb_frame=None, thermal_frame=None):
        return self.process_frames(rgb_frame, thermal_frame)


if __name__ == "__main__":
    engine = dual_spectral_pipeline()
    test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    test_thermal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_thermal, (200, 150), (280, 260), (255, 255, 255), -1)

    results = engine.process_frames(test_rgb, test_thermal)
    print(f"[NeuSE] Pipeline run complete. Detected targets: {len(results)}")