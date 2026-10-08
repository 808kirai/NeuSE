import cv2
import numpy as np
from src.config import config
from src.logger import data_logger


# dual vision processing pipeline simulating mindspore lite execution
class dual_spectral_pipeline:
    def __init__(self):
        self.logger = data_logger()

    def process_frames(self, rgb_frame=None, thermal_frame=None):
        if rgb_frame is None:
            rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if thermal_frame is None:
            thermal_frame = rgb_frame.copy()

        if thermal_frame.ndim == 2:
            thermal_gray = thermal_frame
        else:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        _, heat_mask = cv2.threshold(thermal_gray, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            conf = float(np.clip(0.85 + (area / 10000.0), 0.85, 0.99))

            if conf >= config.conf_threshold:
                peak_heat = int(np.max(thermal_gray[y:y + h, x:x + w]))
                box = [x, y, w, h]
                self.logger.log_event(conf, peak_heat, box)
                hits.append({
                    "bbox": box,
                    "confidence": conf,
                    "thermal_peak": peak_heat,
                })

        return hits


# Compatibility wrapper for the project test suite and older callers.
class VisionPipeline(dual_spectral_pipeline):
    def process_frame(self, rgb_frame=None, thermal_frame=None):
        return self.process_frames(rgb_frame, thermal_frame)


if __name__ == "__main__":
    engine = dual_spectral_pipeline()
    test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    test_thermal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_thermal, (200, 150), (280, 260), (255, 255, 255), -1)

    results = engine.process_frames(test_rgb, test_thermal)
    print(f"pipeline run complete. detected targets: {len(results)}")