import logging
import time
import cv2
import numpy as np
from src.config import config
from src.logger import data_logger

# warnings about skipped frames go to the standard python log
log = logging.getLogger(__name__)

# saved with each detection so results from different detectors can be told apart
detector_name = "classical-baseline"


# finds warm areas in thermal frames
# this is the classical baseline with no trained model behind it
# so it gives no confidence score and does not use the rgb frame yet
class dual_spectral_pipeline:
    def __init__(self):
        # writes detections to the local database
        self.logger = data_logger()
        # reuse clahe instead of building it every frame
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # time spent on the last frame in ms
        self.last_latency_ms = None

    def process_frames(self, rgb_frame=None, thermal_frame=None):
        # main entry point, called once per pair of camera frames
        # returns a list of detections, empty when nothing is found
        # rgb_frame is accepted now so callers stay the same once rgb fusion is added

        # time only the detection so the number reflects the detector itself
        start = time.perf_counter()
        hits = self._detect(thermal_frame)
        self.last_latency_ms = (time.perf_counter() - start) * 1000

        # only touch the database when something was found
        if hits:
            self.logger.log_events(hits, self.last_latency_ms)
        return hits

    def _detect(self, thermal_frame):
        # returns the warm areas in one thermal frame

        # no thermal frame means nothing to detect
        if not isinstance(thermal_frame, np.ndarray) or thermal_frame.size == 0:
            return []

        # turn whatever the camera gives into 8 bit grayscale
        thermal_gray = self._to_gray_uint8(thermal_frame)
        if thermal_gray is None:
            log.warning("skipping thermal frame with unsupported shape %s", thermal_frame.shape)
            return []

        # opencv can still reject odd frames, so skip them instead of crashing
        try:
            # boost contrast to cut background clutter
            enhanced = self.clahe.apply(thermal_gray)

            # keep only the hottest areas
            _, heat_mask = cv2.threshold(enhanced, config.thermal_threshold, 255, cv2.THRESH_BINARY)

            # outline each hot area
            contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error as err:
            log.warning("skipping thermal frame opencv could not process %s", err)
            return []

        # turn each warm area into a detection
        hits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # skip specks and oversized areas
            if area < config.min_blob_area or area > config.max_blob_area:
                continue

            # box around the area as x, y, width and height in pixels
            x, y, w, h = cv2.boundingRect(cnt)
            hits.append({
                "bbox": [x, y, w, h],
                # no model so no confidence to report
                "confidence": None,
                # brightest pixel in the box on a 0 to 255 scale, not a temperature
                "thermal_peak": int(np.max(thermal_gray[y:y + h, x:x + w])),
                "detector": detector_name,
            })

        return hits

    @staticmethod
    def _to_gray_uint8(frame):
        # returns an 8 bit grayscale version of the frame
        # or None when the shape is not one we can read

        # accept grayscale or bgr frames
        # colour frames that are not 8 bit are averaged across channels
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.dtype == np.uint8 else frame.mean(axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        elif frame.ndim == 2:
            gray = frame
        else:
            return None

        if gray.dtype == np.uint8:
            return gray

        # scale 16 bit or float sensor data to 0 to 255
        gray = np.nan_to_num(gray.astype(np.float32))
        low, high = gray.min(), gray.max()
        # a completely flat frame has no warm areas
        if high == low:
            return np.zeros(gray.shape, dtype=np.uint8)
        return ((gray - low) * (255 / (high - low))).astype(np.uint8)


# keeps older callers working
class VisionPipeline(dual_spectral_pipeline):
    def process_frame(self, rgb_frame=None, thermal_frame=None):
        return self.process_frames(rgb_frame, thermal_frame)


# quick check with one fake warm body in an empty scene
# run with python -m src.pipeline
if __name__ == "__main__":
    engine = dual_spectral_pipeline()
    test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    test_thermal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_thermal, (200, 150), (280, 260), (255, 255, 255), -1)

    results = engine.process_frames(test_rgb, test_thermal)
    print(f"[NeuSE] Pipeline run complete. Detected targets: {len(results)} "
          f"in {engine.last_latency_ms:.2f}ms")
