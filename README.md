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

### Live mode

With `--live`, the plot updates as MangoHud writes the log, so you can watch a
session in real time. By default it follows a rolling window of the last 60
seconds (`--window`); pan or zoom to inspect history and the view detaches from
the live edge — press `F` to re-attach. Use `--full-session` to keep the whole
growing session in view instead. Y axes grow to fit new spikes automatically.

```
python mangograph.py --live              # follow the newest log as it's written
python mangograph.py --live --window 30  # 30-second rolling window
```

(Requires a MangoHud build that flushes rows to the CSV during the session
rather than only on stop.)

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
| `F`           | (live mode) Toggle following the live edge            |
| `Q`           | Close the window                                      |

### Options

| Flag            | Effect                                                      |
|-----------------|-------------------------------------------------------------|
| `--show PANELS` | Show only these panels (comma-separated, e.g. `fps,frametime`) |
| `--except PANELS` | Hide these panels from the default set                    |
| `--smooth N`    | Rolling-average window size in frames (default: 30)         |
| `--jitter N`    | Jitter threshold as a multiple of the rolling average (default: 1.5) |
| `--hitch-ms MS` | Frametime at or above which a frame counts as a hitch (default: 100) |
| `--stall-ms MS` | Frametime at or above which a frame counts as a stall (default: 200) |
| `--max-fps N`   | Drop frames at or above N FPS as outliers (default: 1000)   |
| `--keep-all`    | Disable outlier filtering; keep all frames                  |
| `--trim-start SEC` | Drop the first SEC seconds of the session                |
| `--trim-end SEC`   | Drop the last SEC seconds of the session                 |
| `--live`        | Update the plot as the log file is written                  |
| `--window SEC`  | Live: rolling window length (default: 60)                   |
| `--full-session` | Live: show the whole growing session instead of a window   |
| `--poll MS`     | Live: log poll interval in milliseconds (default: 1000)     |
| `-o FILE`       | Save a screenshot to FILE instead of opening a window       |

### Panel Names

Shown top to bottom by default:

| Panel       | Content                                   |
|-------------|-------------------------------------------|
| `load`      | CPU and GPU utilization (%)               |
| `temp`      | CPU and GPU temperature (°C)              |
| `power`     | CPU and GPU power draw (W)                 |
| `memory`    | RAM, VRAM, and swap usage (GB)            |
| `frametime` | Per-frame time (ms) with rolling average  |
| `fps`       | Raw FPS with rolling average, 1% low, avg |

Example — only the fps and frametime panels, smoothed over 60 frames:

```
python mangograph.py --show fps,frametime --smooth 60
```
