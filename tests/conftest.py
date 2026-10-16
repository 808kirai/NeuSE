import os
import sys

import pytest

# add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """give each test its own database so real detections are never touched"""
    # monkeypatch puts the real path back once the test ends
    db = tmp_path / "test.db"
    monkeypatch.setattr(config, "db_path", db)
    return db
