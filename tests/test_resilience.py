import sys
import os
import time
import pytest
import numpy as np

# add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import dual_spectral_pipeline, VisionPipeline


@pytest.fixture
def pipeline():
    """create a pipeline for each test"""
    return dual_spectral_pipeline()


@pytest.fixture
def vision_wrapper():
    """create a legacy pipeline wrapper for each test"""
    return VisionPipeline()


# check valid inputs
def test_valid_frame_processing(pipeline):
    """check normal frame processing"""
    valid_rgb = np.ones((480, 640, 3), dtype=np.uint8) * 128
    valid_thermal = np.ones((480, 640, 3), dtype=np.uint8) * 100

    results = pipeline.process_frames(valid_rgb, valid_thermal)
    assert isinstance(results, list), "Pipeline did not return a list for valid inputs"


# check the older class name still works
def test_legacy_wrapper_compatibility(vision_wrapper):
    """check compatibility with legacy callers"""
    valid_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    results = vision_wrapper.process_frame(valid_frame)
    assert isinstance(results, list), "Legacy wrapper failed to return a list"


# check unusual and invalid inputs
@pytest.mark.parametrize("bad_input", [
    None,                                       # missing frame
    np.zeros((480, 640, 3), dtype=np.uint8),    # black frame
    np.ones((480, 640, 3), dtype=np.uint8) * 255, # fully white frame
    np.ones((1, 1, 3), dtype=np.uint8)           # very small frame
])
def test_pipeline_resilience(pipeline, bad_input):
    """check that unusual inputs do not cause unhandled errors"""
    try:
        results = pipeline.process_frames(bad_input, bad_input)
        assert isinstance(results, list), "Expected list response for adversarial input"
    except Exception as e:
        pytest.fail(f"Pipeline crashed on adversarial input: {e}")


# check processing latency
def test_latency_performance(pipeline):
    """check that frame processing stays within a realistic execution budget"""
    # pure noise is the worst case because it creates many small warm areas
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    thermal = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # warm up the pipeline before measuring
    pipeline.process_frames(rgb, thermal)

    start_time = time.perf_counter()
    pipeline.process_frames(rgb, thermal)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    # adjusted threshold for local CPU testing environment
    assert latency_ms < 150.0, f"Latency warning! Execution took {latency_ms:.2f}ms"