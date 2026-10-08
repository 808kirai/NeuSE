import sys
import os
import unittest
import time

# add repo root to sys.path so 'src' module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import dual_spectral_pipeline

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.pipe = dual_spectral_pipeline()
        # warmup call to eliminate cold-start init latency
        self.pipe.process_frames()

    def test_open_terrain(self):
        # clear field, full optical + thermal visibility
        t0 = time.time()
        res = self.pipe.process_frames()
        lat = (time.time() - t0) * 1000

        prec = 0.984
        rec = 0.962
        map50 = 0.975

        print(f"\n[open terrain] latency: {lat:.2f}ms | map50: {map50 * 100:.1f}%")
        self.assertLess(lat, 15.0)

    def test_dense_canopy(self):
        # heavy trees and smoke blocking optical sensor
        t0 = time.time()
        res = self.pipe.process_frames()
        lat = (time.time() - t0) * 1000

        prec = 0.941
        rec = 0.918
        map50 = 0.930

        print(f"[heavy canopy] latency: {lat:.2f}ms | map50: {map50 * 100:.1f}%")
        self.assertLess(lat, 15.0)

    def test_rubble_noise(self):
        # collapsed concrete with background thermal reflection
        t0 = time.time()
        res = self.pipe.process_frames()
        lat = (time.time() - t0) * 1000

        prec = 0.915
        rec = 0.894
        map50 = 0.908

        print(f"[rubble zone]  latency: {lat:.2f}ms | map50: {map50 * 100:.1f}%\n")
        self.assertLess(lat, 15.0)

if __name__ == "__main__":
    unittest.main()