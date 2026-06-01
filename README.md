# mangograph

An interactive plotter for [MangoHud](https://github.com/flightlessmango/MangoHud)
performance logs. It reads a session's per-frame CSV and draws synchronized
panels for FPS, frametime, CPU/GPU load, temperature, power, and memory, with a
crosshair, span statistics, and OpenGL-accelerated rendering that stays smooth
on multi-hour logs.

## Install

```
pip install -r requirements.txt
```

Requires Python 3, NumPy, pyqtgraph, and a Qt binding (PyQt6 by default).

## Usage

Open the most recent `.csv` in the current directory:

```
python mangograph.py
```

Or pass one or more log files explicitly (each opens in its own window):

```
python mangograph.py MyGame_2026-05-29_12-00-00.csv
```

### Panels

Shown top to bottom by default:

| Panel       | Content                                   |
|-------------|-------------------------------------------|
| `load`      | CPU and GPU utilization (%)               |
| `temp`      | CPU and GPU temperature (°C)              |
| `power`     | CPU and GPU power draw (W)                 |
| `memory`    | RAM, VRAM, and swap usage (GB)            |
| `frametime` | Per-frame time (ms) with rolling average  |
| `fps`       | Raw FPS with rolling average, 1% low, avg |

### Interactions

| Action        | Effect                                                |
|---------------|-------------------------------------------------------|
| Left-drag     | Pan                                                   |
| Scroll wheel  | Zoom the time axis                                    |
| Right-drag    | Select a time span; stats update to that span's range |
| Right-click   | Clear the span selection                              |
| Hover         | Crosshair with per-panel values at the cursor         |
| `Y`           | Toggle the fps/frametime Y axis between fit and full  |
| `L`           | Toggle legend visibility                              |
| `Q`           | Close the window                                      |

### Options

| Flag            | Effect                                                      |
|-----------------|-------------------------------------------------------------|
| `--show PANELS` | Show only these panels (comma-separated, e.g. `fps,frametime`) |
| `--except PANELS` | Hide these panels from the default set                    |
| `--smooth N`    | Rolling-average window size in frames (default: 30)         |
| `--stutter N`   | Stutter threshold as a multiple of the rolling average (default: 1.5) |
| `--max-fps N`   | Drop frames at or above N FPS as outliers (default: 1000)   |
| `--keep-all`    | Disable outlier filtering; keep all frames                  |
| `--trim-start SEC` | Drop the first SEC seconds of the session                |
| `--trim-end SEC`   | Drop the last SEC seconds of the session                 |
| `-o FILE`       | Save a screenshot to FILE instead of opening a window       |

Example — only the fps and frametime panels, smoothed over 60 frames:

```
python mangograph.py --show fps,frametime --smooth 60
```
