import cv2
import numpy as np
from src.config import config
from src.logger import data_logger

# dual vision processing pipeline simulating mindspore lite execution
class dual_spectral_pipeline:
    def __init__(self):
        self.logger = data_logger()

    def process_frames(self, rgb_frame, thermal_frame):
        # drop thermal frame to single channel for fast heat thresholding
        thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        _, heat_mask = cv2.threshold(thermal_gray, 200, 255, cv2.THRESH_BINARY)
        
        # isolate high heat contours
        contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hits = []
        for cnt in contours:
            # ignore minor sensor noise or tiny glints
            if cv2.contourArea(cnt) < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # calculate baseline confidence score
            conf = float(np.clip(0.85 + (cv2.contourArea(cnt) / 10000.0), 0.85, 0.99))

            if conf >= config.conf_threshold:
                # get highest intensity point inside bounding region
                peak_heat = int(np.max(thermal_gray[y:y+h, x:x+w]))
                box = [x, y, w, h]
                
                # log locally immediately
                self.logger.log_event(conf, peak_heat, box)
                hits.append({
                    "bbox": box,
                    "confidence": conf,
                    "thermal_peak": peak_heat
                })

        return hits

if __name__ == "__main__":
    # quick sanity check with dummy blank feeds
    engine = dual_spectral_pipeline()
    test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    test_thermal = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # draw simulated hot spot in center
    cv2.rectangle(test_thermal, (200, 150), (280, 260), (255, 255, 255), -1)
    
    results = engine.process_frames(test_rgb, test_thermal)
    print(f"pipeline run complete. detected targets: {len(results)}")