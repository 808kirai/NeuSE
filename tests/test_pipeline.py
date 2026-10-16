import sys
import os
import statistics
import unittest
import time
import cv2
import numpy as np

# add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import dual_spectral_pipeline

# robustness and speed checks written with unittest
# pytest runs these too and gives them their own test database
class TestPipelineRobustness(unittest.TestCase):

    def setUp(self):
        self.pipe = dual_spectral_pipeline()
        # warm up the pipeline before testing
        self.pipe.process_frames()

    # check that invalid inputs are handled
    def test_corrupted_and_empty_inputs(self):
        # check missing frames
        res_null = self.pipe.process_frames(None, None)
        self.assertIsInstance(res_null, list)

        # check empty arrays
        empty_arr = np.array([])
        res_empty = self.pipe.process_frames(empty_arr, empty_arr)
        self.assertIsInstance(res_empty, list)

        # check unsupported frame shapes
        odd_shape = np.zeros((10, 10, 1), dtype=np.uint8)
        res_odd = self.pipe.process_frames(odd_shape, odd_shape)
        self.assertIsInstance(res_odd, list)

        # check a dark scene, which should give no detections
        pitch_black = np.zeros((480, 640, 3), dtype=np.uint8)
        res_black = self.pipe.process_frames(pitch_black, pitch_black)
        self.assertEqual(len(res_black), 0)

    # test detection in a cluttered scene
    def test_rubble_clutter_scene(self):
        # generate noisy rubble-like background
        rgb_rubble = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        thermal_rubble = np.random.randint(0, 80, (480, 640, 3), dtype=np.uint8)
        
        # add a simulated thermal target
        cv2.rectangle(thermal_rubble, (200, 150), (260, 250), (255, 255, 255), -1)

        res = self.pipe.process_frames(rgb_rubble, thermal_rubble)
        self.assertGreaterEqual(len(res), 1)

    # measure full stream processing latency
    def test_e2e_stream_latency(self):
        # encode a frame as a compressed stream
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded_buffer = cv2.imencode('.jpg', raw_frame)

        # time many runs and use the median so one slow run from background work does not fail the test
        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            # decode the frame before processing
            decoded_rgb = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
            self.pipe.process_frames(decoded_rgb, decoded_rgb)
            timings.append((time.perf_counter() - t0) * 1000)
        e2e_lat = statistics.median(timings)

        # print the time so it shows up with pytest -s
        print(f"\n[E2E Stream Profile] Ingestion + Decode + Processing Latency: {e2e_lat:.2f}ms median of 20 runs")
        self.assertLess(e2e_lat, 15.0)

if __name__ == "__main__":
    unittest.main()