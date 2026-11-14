import logging
import sqlite3
import time
from collections import deque

import cv2
import numpy as np
from src.config import config
from src.geo import ground_position
from src.logger import data_logger, now
from src.person_check import person_check

# warnings about skipped frames go to the standard python log
log = logging.getLogger(__name__)

# saved with each detection so results from different detectors can be told apart
# these are the same names used to choose a detector
detector_name = "warm-spots"
person_check_name = "person-check"
detectors = (person_check_name, detector_name)

# frame size the person check was trained on, width and height
trained_size = (640, 512)
# most detections held in memory while the database cannot be written
max_unsaved = 5000


# finds people in thermal frames with one of two classical detectors
# person-check scores small warm spots with a small trained check and gives a confidence
# warm-spots flags any bright area of the right size and gives no confidence
# the rgb frame is accepted but not used yet
class thermal_pipeline:
    def __init__(self, detector=None):
        # writes detections to the local database
        self.logger = data_logger()
        # reuse clahe instead of building it every frame
        grid = config.clahe_tile_grid
        self.clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(grid, grid))
        # time spent on the last frame in ms, and on recent frames for the average
        self.last_latency_ms = None
        self.recent_ms = deque(maxlen=100)
        # frames seen, and when the status row was last written
        self.frames = 0
        # minus infinity so the first frame always writes one
        self.last_status = float("-inf")
        # detections waiting for a database that could not be written
        self.unsaved = []
        # so the frame size warning shows once
        self.warned_about_size = False

        # pick the detector, the person check needs its trained weights
        self.detector = detector or config.detector
        if self.detector not in detectors:
            raise ValueError(f"unknown detector {self.detector}, use one of {detectors}")
        self.checker = None
        if self.detector == person_check_name:
            if config.person_check_path.exists():
                self.checker = person_check(config.person_check_path, config.max_candidates)
            else:
                # keep detecting with the older method instead of stopping the unit
                log.warning("no weights at %s, using warm-spots", config.person_check_path)
                self.detector = detector_name

    def process_frames(self, rgb_frame=None, thermal_frame=None, position=None, pose=None):
        # main entry point, called once per pair of camera frames
        # returns a list of detections, empty when nothing is found
        # rgb_frame is accepted now so callers stay the same once rgb fusion is added
        # position is the unit's latitude and longitude for this frame, the configured one when not given
        # pose is the unit's altitude in metres above ground, heading in degrees from north and camera tilt
        # in degrees below level, when given each sighting is placed on the ground instead of at the unit

        # time only the detection so the number reflects the detector itself
        start = time.perf_counter()
        hits = self._detect(thermal_frame)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        self.recent_ms.append(self.last_latency_ms)

        if hits:
            self._place(hits, thermal_frame, position, pose)
        self._save(hits, position)

        # a short status note every few seconds, not every frame, so the disk is not kept busy
        self.frames += 1
        if time.monotonic() - self.last_status >= config.status_every:
            try:
                self.logger.note_status(position, self.detector, self.frames,
                                        sum(self.recent_ms) / len(self.recent_ms))
            except sqlite3.Error as err:
                log.warning("could not save the status note, %s", err)
            self.last_status = time.monotonic()
        return hits

    def _place(self, hits, frame, position, pose):
        # gives each hit a position, on the ground when the pose allows it and at the unit otherwise
        unit = position or (config.default_lat, config.default_lon)
        height, width = frame.shape[:2]
        for hit in hits:
            ground = None
            if pose is not None:
                ground = ground_position(unit, pose, hit["bbox"], (width, height), config.camera_fov,
                                         config.max_ground_range)
            hit["position"] = ground or unit
            hit["position_source"] = "projected" if ground else "unit"

    def _save(self, hits, position):
        # writes detections to the database, and holds them in memory when it cannot be written
        # a full disk or a locked file must not stop the unit detecting
        if hits:
            self.unsaved.append((hits, self.last_latency_ms, position, now()))
        while self.unsaved:
            try:
                self.logger.log_events(*self.unsaved[0])
            except sqlite3.Error as err:
                held = sum(len(batch[0]) for batch in self.unsaved)
                log.error("could not save detections, holding %s in memory and trying again next frame, %s", held, err)
                # keep memory bounded, the oldest go first
                while sum(len(batch[0]) for batch in self.unsaved) > max_unsaved and len(self.unsaved) > 1:
                    dropped = self.unsaved.pop(0)
                    log.error("dropped %s detections that could not be saved", len(dropped[0]))
                return
            self.unsaved.pop(0)

    def _detect(self, thermal_frame, min_score=None):
        # returns the people found in one thermal frame
        # min_score overrides the person check threshold, scoring uses 0 to keep every candidate

        # no thermal frame means nothing to detect
        if not isinstance(thermal_frame, np.ndarray) or thermal_frame.size == 0:
            return []

        # turn whatever the camera gives into 8 bit grayscale
        thermal_gray = self._to_gray_uint8(thermal_frame)
        if thermal_gray is None:
            log.warning("skipping thermal frame with unsupported shape %s", thermal_frame.shape)
            return []

        if self.checker is not None:
            return self._person_check_hits(thermal_gray, min_score)

        # opencv can still reject odd frames, so skip them instead of crashing
        try:
            # boost contrast to cut background clutter
            enhanced = self.clahe.apply(thermal_gray)

            # keep only the hottest areas
            _, heat_mask = cv2.threshold(enhanced, config.thermal_threshold, 255, cv2.THRESH_BINARY)

            # label each hot area with its box and pixel count in one pass
            # much faster than outlining every speck on noisy frames, which took about 140 ms
            _, _, stats, _ = cv2.connectedComponentsWithStats(heat_mask, connectivity=8)
        except cv2.error as err:
            log.warning("skipping thermal frame opencv could not process %s", err)
            return []

        # row 0 is the background, the rest are warm areas
        stats = stats[1:]
        areas = stats[:, cv2.CC_STAT_AREA]
        # skip specks and oversized areas
        keep = stats[(areas >= config.min_blob_area) & (areas <= config.max_blob_area)]

        # turn each warm area into a detection
        hits = []
        for x, y, w, h, _ in keep.tolist():
            # box around the area as x, y, width and height in pixels
            hits.append({
                "bbox": [x, y, w, h],
                # no model so no confidence to report
                "confidence": None,
                # brightest pixel in the box on a 0 to 255 scale, not a temperature
                "thermal_peak": int(np.max(thermal_gray[y:y + h, x:x + w])),
                "detector": detector_name,
            })

        return hits

    def _person_check_hits(self, thermal_gray, min_score):
        # runs the person check and turns its boxes into detections
        height, width = thermal_gray.shape

        # resize so people are about the size the check was trained on, boxes are scaled back below
        scale = config.frame_scale
        work = thermal_gray
        if scale != 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            work = cv2.resize(thermal_gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if not self.warned_about_size and (work.shape[1], work.shape[0]) != trained_size:
            log.warning("the person check was trained on %s by %s frames where a person is about 15 pixels tall, "
                        "this frame is %s by %s, set NEUSE_FRAME_SCALE if people look much bigger or smaller",
                        *trained_size, work.shape[1], work.shape[0])
            self.warned_about_size = True

        # opencv or numpy can still reject odd frames, so skip them instead of crashing
        try:
            found = self.checker.detect(work, min_score)
        except (cv2.error, ValueError, MemoryError) as err:
            log.warning("skipping thermal frame the person check could not process %s", err)
            return []

        hits = []
        for (x, y, w, h), score in found:
            x, y, w, h = x / scale, y / scale, w / scale, h / scale
            # fitted boxes can reach past the frame edge, so trim them to whole pixels inside it
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
            if x1 <= x0 or y1 <= y0:
                continue
            hits.append({
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                # chance this spot is a person, 0 to 1, calibrated on the validation images
                "confidence": float(score),
                "thermal_peak": int(np.max(thermal_gray[y0:y1, x0:x1])),
                "detector": person_check_name,
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
        gray = np.nan_to_num(gray.astype(np.float32), posinf=0.0, neginf=0.0)

        # a fixed range keeps a cold scene dark instead of stretching it
        if config.sensor_range is not None:
            low, high = config.sensor_range
            scaled = (gray - low) * (255 / (high - low))
            return np.clip(scaled, 0, 255).astype(np.uint8)

        # take the range from a median filtered copy so one dead pixel cannot set it
        # without this a single stuck hot pixel squashes a real person below the threshold
        smooth = cv2.medianBlur(gray, 3)
        low, high = float(smooth.min()), float(smooth.max())
        # a completely flat frame has no warm areas
        if high <= low:
            return np.zeros(gray.shape, dtype=np.uint8)
        scaled = (gray - low) * (255 / (high - low))
        return np.clip(scaled, 0, 255).astype(np.uint8)


# the first name of the class, kept so older callers still work
# it only ever used the thermal frame, so the new name says what it does
dual_spectral_pipeline = thermal_pipeline


# keeps older callers working
class VisionPipeline(thermal_pipeline):
    def process_frame(self, rgb_frame=None, thermal_frame=None):
        return self.process_frames(rgb_frame, thermal_frame)


# quick check with one fake warm body in an empty scene
# run with python -m src.pipeline
if __name__ == "__main__":
    # a big made up warm shape, so this uses the warm-spots detector
    engine = thermal_pipeline(detector_name)
    test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    test_thermal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_thermal, (200, 150), (280, 260), (255, 255, 255), -1)

    results = engine.process_frames(test_rgb, test_thermal)
    print(f"[NeuSE] Pipeline run complete. Detected targets: {len(results)} "
          f"in {engine.last_latency_ms:.2f}ms")
