# NeuSE

Neural Sentinel Edge helps rescue teams find survivors in thermal camera footage, even where there is no network. Each field unit runs detection on the spot and is designed to send its results to a cloud command center whenever a connection is available.

Built for the Huawei ICT Competition 2026–2027 Innovation Track.

## Why it matters
After earthquakes and floods, or in remote areas, the network is often the first thing to go. AI tools that depend on the cloud stop working just when rescuers need them most. NeuSE keeps detection on the device, so it works offline, and is designed to share results with the wider team once a link is back.

## Where it stands
NeuSE is an early prototype.

**Working now**
- Thermal detection using a classical computer vision baseline
- Safe handling of missing, empty, 16 bit, float and odd shaped frames
- A local detection log, with the measured processing time of every frame
- A local dashboard showing logged detections
- Automated tests

**Planned**
- A person detector trained with MindYOLO on Huawei Cloud ModelArts, using the HIT-UAV thermal dataset
- On-device inference with MindSpore Lite
- Using the RGB camera to confirm thermal hits and cut false alarms
- Syncing detections to a command center on Huawei Cloud
- Live GPS, in place of the fixed position each unit uses today

The baseline flags any warm area of the right size, so it cannot tell a person from a car engine. It gives no confidence score because there is no model behind it. It stays in the project as the benchmark the trained detector has to beat.

## Layout
| Path | What it holds |
|---|---|
| `src/pipeline.py` | Thermal detection |
| `src/logger.py` | Local detection log in SQLite |
| `src/config.py` | Thresholds, unit position and database location |
| `dashboard/` | Local web dashboard |
| `scripts/simulate_feed.py` | Sends generated thermal frames through the real pipeline for demos |
| `models/` | Where trained model files will go |
| `tests/` | Automated tests |
| `docs/ARCHITECTURE.md` | How the parts fit together |

## Run it
Tested on Python 3.12.

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt

# run the tests
pytest -v

# terminal 1: generate detections
python scripts/simulate_feed.py --frames 120

# terminal 2: open the dashboard at http://127.0.0.1:5000
python dashboard/app.py
```

To run the pipeline once on a test frame, use `python -m src.pipeline`.

## Settings
| Variable | Default | Use |
|---|---|---|
| `NEUSE_DB` | `neuse_local.db` in the project folder | Detection database |
| `NEUSE_HOST` | `127.0.0.1` | Set to `0.0.0.0` to open the dashboard to other devices |
| `NEUSE_PORT` | `5000` | Dashboard port |
| `NEUSE_DEBUG` | off | Set to `1` for Flask debug mode, on a trusted dev machine only |

## Performance
Every frame is timed, and the time is saved with its detections. The dashboard shows the average. Run `pytest -s` to see the numbers on your own machine.

## License
MIT. See `LICENSE`.
