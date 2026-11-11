import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from simulate_feed import synthetic_thermal_frame, to_raw16
from src.pipeline import dual_spectral_pipeline


def test_demo_frames_show_both_targets():
    rng = np.random.default_rng(0)
    frame = synthetic_thermal_frame(0, rng)
    assert len(dual_spectral_pipeline().process_frames(thermal_frame=frame)) == 2


def test_raw16_frames_with_dead_pixels_still_show_both_targets():
    rng = np.random.default_rng(0)
    raw = to_raw16(synthetic_thermal_frame(0, rng), rng, dead_pixels=3)
    assert raw.dtype == np.uint16
    assert (raw == 65535).sum() >= 1
    assert len(dual_spectral_pipeline().process_frames(thermal_frame=raw)) == 2
