#!/usr/bin/env python3
"""MangoHud log plotter — pyqtgraph."""

import io
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

_RIGHT = QtCore.Qt.MouseButton.RightButton

ALL_PANELS = ["load", "temp", "power", "memory", "frametime", "fps"]

C = {
    "fps_raw":  (80,  80,  80),
    "fps_avg":  (0,   191, 255),
    "ft_avg":   (255, 111, 145),
    "pct_line": (255, 215, 0),
    "low_line": (255, 107, 107),
    "cpu":      (255, 153, 0),
    "gpu":      (0,   230, 118),
    "ram":      (41,  182, 246),
    "vram":     (171, 71,  188),
    "swap":     (239, 154, 154),
    "fg":       (204, 204, 204),
    "bg":       (26,  26,  26),
    "stats_bg": (34,  34,  34, 210),
    "span":     (255, 255, 255, 25),
    "cross":    (200, 200, 200, 100),
}

# Global config must be set before any widgets are created.
pg.setConfigOptions(background=pg.mkColor(*C["bg"]), foreground=pg.mkColor(*C["fg"]),
                    useOpenGL=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _hms(seconds):
    h, r = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(r, 60)
    return h, m, s


def fmt_duration(seconds):
    h, m, s = _hms(seconds)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"


def rolling_avg(x, window):
    # Centered moving average. Divide by the count of samples actually overlapping
    # the window at each position (fewer at the array ends) rather than by `window`,
    # so the curve isn't dragged toward zero at the edges — which in live mode
    # would otherwise droop right at the live edge on every frame.
    window = min(window, len(x))
    kernel = np.ones(window)
    sums   = np.convolve(x, kernel, mode="same")
    counts = np.convolve(np.ones(len(x)), kernel, mode="same")
    return sums / counts


def _num(s):
    """Wrap a stat value in bold white so it stands out from its muted label
    in the (HTML) stat boxes."""
    return f'<b style="color:#fff">{s}</b>'


def load_log(path):
    with open(path) as f:
        meta    = dict(zip(f.readline().strip().split(","), f.readline().strip().split(",")))
        headers = f.readline().strip().split(",")
        data    = np.loadtxt(f, delimiter=",")
    return meta, data, headers


class LogTail:
    """Incrementally reads rows appended to a MangoHud CSV while it is still
    being written. Parses the two metadata lines and the header once, then each
    read_rows() call returns the complete rows appended since the last call
    (buffering any trailing partial line for next time)."""

    def __init__(self, path):
        self.f       = open(path)
        self.meta    = dict(zip(self.f.readline().strip().split(","),
                                self.f.readline().strip().split(",")))
        self.headers = self.f.readline().strip().split(",")
        self._buf    = ""

    def read_rows(self):
        chunk = self.f.read()
        if chunk:
            self._buf += chunk
        if "\n" not in self._buf:
            return None
        body, _, self._buf = self._buf.rpartition("\n")
        body = body.strip("\n")
        if not body:
            return None
        try:
            rows = np.loadtxt(io.StringIO(body), delimiter=",")
        except ValueError:
            # Likely a torn write; keep the data and retry once more arrives.
            self._buf = body + "\n" + self._buf
            return None
        return rows.reshape(1, -1) if rows.ndim == 1 else rows


def _await_initial(tail, timeout=10.0):
    """Block briefly until the log has at least a couple of rows to plot."""
    rows = tail.read_rows()
    waited = 0.0
    while (rows is None or len(rows) < 2) and waited < timeout:
        time.sleep(0.2)
        waited += 0.2
        more = tail.read_rows()
        if more is not None:
            rows = more if rows is None else np.concatenate([rows, more])
    if rows is None or len(rows) < 2:
        print("Live: not enough data in the log yet — is MangoHud logging?")
        sys.exit(1)
    return rows


# LOD pyramid: cap on how many source points a curve renders within the viewport,
# and the decimation step between successive levels. update_lod() picks the finest
# level whose in-window count stays under the cap, so the rendered detail stays in
# the (cap/RATIO, cap] band at every zoom — always crisp, never re-scanning the
# whole series. The cap comfortably exceeds the pixel width, and pyqtgraph's own
# autoDownsample reduces the rest to viewport resolution.
LOD_MAX_VISIBLE = 16_000
LOD_RATIO       = 4

# Y-axis padding. The default ("fit") view frames the data on both ends —
# fit_range pads the min..max span by FIT_PAD so the lower bound tracks the
# data, not zero. The full view (Y toggle, fps/frametime only) stays anchored at
# zero and hugs the absolute max by Y_PAD_FULL.
FIT_PAD    = 0.1
Y_PAD_FULL = 1.05


def fit_range(lo, hi):
    """Pad [lo, hi] by a fraction of its span for the fit view, with a small
    floor so a near-flat series still has visible height. Clamped at zero since
    these metrics are non-negative."""
    margin = max((hi - lo) * FIT_PAD, abs(hi) * 0.02, 1e-6)
    return (max(0.0, lo - margin), hi + margin)


def build_pyramid(x, y, cap=LOD_MAX_VISIBLE, ratio=LOD_RATIO):
    """Build min/max LOD levels for a series, finest (full) first. Each coarser
    level peak-decimates the full data by a growing factor so spikes survive at
    every level; the series is cast to float32 to halve memory and GL upload."""
    y = np.asarray(y, dtype=np.float32)
    levels = [(x, y)]
    f = ratio
    while (len(y) // f) >= 2:
        n2 = (len(y) // f) * f
        xs = x[:n2].reshape(-1, f)[:, 0]
        ys = y[:n2].reshape(-1, f)
        ox = np.repeat(xs, 2)
        oy = np.empty(ys.shape[0] * 2, dtype=np.float32)
        oy[0::2] = ys.min(axis=1)
        oy[1::2] = ys.max(axis=1)
        levels.append((ox, oy))
        if len(oy) <= cap:
            break
        f *= ratio
    return levels


def lod_plot(p, x, y, pen, name=None, registry=None):
    """Plot a series with a tiered LOD pyramid. Shows the coarsest level initially
    (full zoom-out); update_lod() swaps the active level on zoom. pyqtgraph's
    clipToView + autoDownsample then slice/reduce the active level per frame.

    Options must be set via the opts dict after the item is in the scene to avoid
    a pyqtgraph parent-chain AttributeError."""
    levels = build_pyramid(x, y)
    idx = len(levels) - 1
    li = p.plot(*levels[idx], pen=pen, name=name, skipFiniteCheck=True)
    li.opts['clipToView']       = True
    li.opts['autoDownsample']   = True
    li.opts['downsampleMethod'] = 'peak'
    if registry is not None and len(levels) > 1:
        registry.append({"item": li, "levels": levels, "idx": idx})
    return li


# ── custom widgets ────────────────────────────────────────────────────────────

class TimeAxisItem(pg.AxisItem):
    def __init__(self, total_seconds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._show_hours = total_seconds >= 3600

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            h, m, s = _hms(v)
            out.append(f"{h}:{m:02d}:{s:02d}" if self._show_hours else f"{m}:{s:02d}")
        return out


class SpanViewBox(pg.ViewBox):
    """ViewBox with right-drag span selection instead of zoom rectangle."""
    sigSpanDrag     = QtCore.Signal(float, float)
    sigSpanFinished = QtCore.Signal(float, float)
    sigSpanCleared  = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._span_start = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == _RIGHT:
            ev.accept()
            x = self.mapToView(ev.pos()).x()
            if ev.isStart():
                self._span_start = self.mapToView(ev.buttonDownPos()).x()
            if self._span_start is not None:
                lo = min(self._span_start, x)
                hi = max(self._span_start, x)
                if ev.isFinish():
                    if hi - lo > 0.1:
                        self.sigSpanFinished.emit(lo, hi)
                    else:
                        self.sigSpanCleared.emit()
                    self._span_start = None
                else:
                    self.sigSpanDrag.emit(lo, hi)
        else:
            super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.button() == _RIGHT:
            ev.accept()
            self.sigSpanCleared.emit()
        else:
            super().mouseClickEvent(ev)


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(path, smooth=30, jitter_threshold=1.5, show=None, max_fps=1000.0,
         trim_start=0.0, trim_end=0.0, hitch_ms=100.0, stall_ms=200.0,
         live=False, full_session=False, follow_window=60.0, poll_ms=1000):
    tail = None
    if live:
        tail = LogTail(path)
        data = _await_initial(tail)
        meta, headers = tail.meta, tail.headers
    else:
        meta, data, headers = load_log(path)
    col = {name: i for i, name in enumerate(headers)}

    def filter_fps(rows, announce=False):
        if max_fps is None:
            return rows
        keep    = rows[:, col["fps"]] < max_fps
        dropped = len(rows) - int(keep.sum())
        if announce and dropped:
            print(f"Dropped {dropped} of {len(rows)} rows "
                  f"({dropped / len(rows) * 100:.2f}%) at >= {max_fps:g} FPS")
        return rows[keep]

    data = filter_fps(data, announce=True)

    # Trimming is a post-hoc operation; it has no meaning for a growing log.
    if (trim_start or trim_end) and not live:
        rel  = (data[:, col["elapsed"]] - data[0, col["elapsed"]]) / 1e9
        keep = (rel >= trim_start) & (rel <= rel[-1] - trim_end)
        if keep.sum() < 2:
            print(f"Trim removes all but {int(keep.sum())} of {len(data)} rows "
                  f"(session is {fmt_duration(rel[-1])}); skipping trim")
        else:
            print(f"Trimmed {len(data) - int(keep.sum())} rows "
                  f"(-{trim_start:g}s start, -{trim_end:g}s end)")
            data = data[keep]

    # All data-derived arrays/scalars are plot() locals so the per-panel
    # slice_fn/hover_fn/y closures read them by name; recompute() rebinds them
    # (via nonlocal) after live rows are appended and every consumer sees the
    # new data automatically. The elapsed anchor is fixed at the first row.
    elapsed0 = float(data[0, col["elapsed"]])
    t = fps = ft_ms = cpu_load = gpu_load = cpu_temp = gpu_temp = None
    gpu_power = cpu_power = ram_used = gpu_vram = swap_used = None
    fps_avg = ft_roll = full_mask = None
    fps_mean = fps_p1 = ft_mean = ft_p99 = t0 = t_end = 0.0

    def recompute():
        nonlocal t, fps, ft_ms, cpu_load, gpu_load, cpu_temp, gpu_temp
        nonlocal gpu_power, cpu_power, ram_used, gpu_vram, swap_used
        nonlocal fps_avg, ft_roll, full_mask
        nonlocal fps_mean, fps_p1, ft_mean, ft_p99, t0, t_end
        t         = (data[:, col["elapsed"]] - elapsed0) / 1e9
        fps       = data[:, col["fps"]]
        ft_ms     = data[:, col["frametime"]]
        cpu_load  = data[:, col["cpu_load"]]
        gpu_load  = data[:, col["gpu_load"]]
        cpu_temp  = data[:, col["cpu_temp"]]
        gpu_temp  = data[:, col["gpu_temp"]]
        gpu_power = data[:, col["gpu_power"]]
        cpu_power = data[:, col["cpu_power"]]
        ram_used  = data[:, col["ram_used"]]
        gpu_vram  = data[:, col["gpu_vram_used"]]
        swap_used = data[:, col["swap_used"]]
        fps_avg   = rolling_avg(fps, smooth)
        ft_roll   = rolling_avg(ft_ms, smooth)
        fps_mean  = float(np.mean(fps))
        fps_p1    = float(np.percentile(fps, 1))
        ft_mean   = float(np.mean(ft_ms))
        ft_p99    = float(np.percentile(ft_ms, 99))
        full_mask = np.ones(len(t), dtype=bool)
        t0, t_end = float(t[0]), float(t[-1])

    recompute()
    has_cpu_power = bool(cpu_power.any())

    if show is None:
        show = set(ALL_PANELS)
    active = [p for p in ALL_PANELS if p in show]
    if not active:
        return None, ()

    game  = Path(path).stem
    title = f"{game}  |  {meta.get('gpu', '')}  |  {meta.get('cpu', '')}"

    # ── window ──
    win = pg.GraphicsLayoutWidget(title=title, show=False)
    win.resize(1400, max(500, sum(200 if p == "fps" else 130 for p in active)))
    win.setWindowTitle(title)
    win.setBackground(pg.mkColor(*C["bg"]))
    win.ci.layout.setContentsMargins(8, 8, 8, 8)
    win.ci.layout.setSpacing(2)

    dash       = QtCore.Qt.PenStyle.DashLine
    font_stats = QtGui.QFont("monospace", 8)
    cross_pen  = pg.mkPen(C["cross"], width=0.8)

    def make_legend(p):
        return p.addLegend(
            offset=(-10, 10),
            labelTextColor=pg.mkColor(*C["fg"]),
            labelTextSize="8pt",
            pen=pg.mkPen(None),
            brush=pg.mkBrush(*C["stats_bg"]),
        )

    def make_text_item(p, anchor=(0, 0), visible=True, overlay=False):
        item = pg.TextItem(anchor=anchor, color=pg.mkColor(*C["fg"]))
        item.setFont(font_stats)
        item.fill = pg.mkBrush(*C["stats_bg"])
        if not visible:
            item.setVisible(False)
        if overlay:
            # Parent to PlotItem (pixel space) so the item doesn't move during pan/zoom.
            item.setParentItem(p)
            item.setZValue(100)
        else:
            p.addItem(item, ignoreBounds=True)
        return item

    def make_region(p):
        r = pg.LinearRegionItem(
            brush=pg.mkBrush(*C["span"]), movable=False,
            pen=pg.mkPen(None),
        )
        r.setVisible(False)
        p.addItem(r)
        return r

    # ── build one PlotItem per active panel ──
    panel_states = []
    first_p      = None
    lod_curves   = []
    live_curves  = []   # (curve item, get_y) updated each live tick
    live_lines   = []   # (reference InfiniteLine, get_value)

    def _lp(p, x, y, pen, name=None, get=None):
        # Live mode skips the LOD pyramid (rows accrue slowly) and plots a plain
        # curve that on_tick re-feeds via setData; static mode builds the pyramid.
        if live:
            it = p.plot(x, y, pen=pen, name=name, skipFiniteCheck=True)
            it.opts['clipToView']       = True
            it.opts['autoDownsample']   = True
            it.opts['downsampleMethod'] = 'peak'
            if get is not None:
                live_curves.append((it, get))
            return it
        return lod_plot(p, x, y, pen, name=name, registry=lod_curves)

    def _ref_line(p, get, color):
        ln = p.addLine(y=get(), pen=pg.mkPen(color, width=1, style=dash))
        if live:
            live_lines.append((ln, get))
        return ln

    for row_i, name in enumerate(active):
        is_last = (row_i == len(active) - 1)
        vb = SpanViewBox()
        p  = win.addPlot(
            row=row_i, col=0, viewBox=vb,
            axisItems={"bottom": TimeAxisItem(t_end, orientation="bottom")},
        )
        if not is_last:
            p.getAxis("bottom").setStyle(showValues=False)
            p.getAxis("bottom").setHeight(0)
        else:
            p.setLabel("bottom", f"[{fmt_duration(t_end)}]")
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setMenuEnabled(False)
        p.getAxis("left").setWidth(55)
        for side in ("top", "right"):
            p.showAxis(side)
            p.getAxis(side).setStyle(showValues=False, tickLength=0)
        vb.setMouseEnabled(x=True, y=False)
        vb.disableAutoRange('y')
        win.ci.layout.setRowStretchFactor(row_i, 2 if name == "fps" else 1)

        if first_p is None:
            first_p = p
        else:
            p.setXLink(first_p)

        lg = make_legend(p)

        # ── per-panel data lines, stats, and y-range ──
        if name == "fps":
            p.setLabel("left", "FPS")
            _lp(p, t, fps,     pg.mkPen(C["fps_raw"], width=0.8), "raw", get=lambda: fps)
            _lp(p, t, fps_avg, pg.mkPen(C["fps_avg"], width=1.5), f"{smooth}-frame avg", get=lambda: fps_avg)
            _ref_line(p, lambda: fps_p1,   C["low_line"])
            _ref_line(p, lambda: fps_mean, C["pct_line"])
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["pct_line"], width=1, style=dash)), "avg")
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["low_line"], width=1, style=dash)), "1% low")
            y_fn  = lambda sl: fit_range(float(fps_avg[sl].min()), float(fps_avg[sl].max()))
            yf_fn = lambda sl: (0, float(fps[sl].max()) * Y_PAD_FULL)
            def slice_fn(mask):
                s = fps[mask]
                return "  ".join([
                    f"avg {_num(f'{np.mean(s):.0f}')}",
                    f"1% low {_num(f'{np.percentile(s,1):.0f}')}",
                    f"0.1% low {_num(f'{np.percentile(s,0.1):.0f}')}",
                    f"max {_num(f'{s.max():.0f}')}",
                ])
            hover_fn = lambda i: f"t {fmt_duration(float(t[i]))}    FPS {fps[i]:.0f}"

        elif name == "frametime":
            p.setLabel("left", "Frametime (ms)")
            _lp(p, t, ft_ms,   pg.mkPen(C["fps_raw"], width=0.8), "raw", get=lambda: ft_ms)
            _lp(p, t, ft_roll, pg.mkPen(C["ft_avg"],  width=1.5), f"{smooth}-frame avg", get=lambda: ft_roll)
            _ref_line(p, lambda: ft_mean, C["pct_line"])
            _ref_line(p, lambda: ft_p99,  C["low_line"])
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["pct_line"], width=1, style=dash)), "avg")
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["low_line"], width=1, style=dash)), "99th pct")
            y_fn  = lambda sl: fit_range(float(ft_roll[sl].min()), float(ft_roll[sl].max()))
            yf_fn = lambda sl: (0, float(ft_ms[sl].max()) * Y_PAD_FULL)
            def slice_fn(mask):
                ft_s = ft_ms[mask]
                n    = len(ft_s)
                # jitter: small relative unevenness (frame exceeds local rolling
                # avg by the threshold factor), as a % of frames. reuses the
                # precomputed rolling avg rather than re-convolving the slice.
                jit  = (ft_s > jitter_threshold * ft_roll[mask]).sum() / n * 100 if n else 0.0
                # hitches/stalls: absolute long frames, perception-anchored in ms.
                hits = int(((ft_s >= hitch_ms) & (ft_s < stall_ms)).sum())
                st_m = ft_s >= stall_ms
                stl  = int(st_m.sum())
                froz = float(ft_s[st_m].sum()) / 1000  # total stall time, ms -> s
                return "  ".join([
                    f"avg {_num(f'{np.mean(ft_s):.1f} ms')}",
                    f"99th {_num(f'{np.percentile(ft_s, 99):.1f} ms')}",
                    f"jitter {_num(f'{jit:.1f}%')}",
                    f"hitches ({hitch_ms:g}ms+) {_num(hits)}",
                    f"stalls ({stall_ms:g}ms+) {_num(stl)} ({_num(f'{froz:.1f}s')})",
                ])
            hover_fn = lambda i: f"{ft_ms[i]:.2f} ms  (avg {ft_roll[i]:.2f})"

        elif name == "load":
            p.setLabel("left", "Load (%)")
            _lp(p, t, cpu_load, pg.mkPen(C["cpu"], width=1), "CPU", get=lambda: cpu_load)
            _lp(p, t, gpu_load, pg.mkPen(C["gpu"], width=1), "GPU", get=lambda: gpu_load)
            y_fn = yf_fn = lambda sl: (0, 105)
            def slice_fn(mask):
                cl, gl = cpu_load[mask], gpu_load[mask]
                return "  ".join([
                    f"CPU  avg {_num(f'{np.mean(cl):.0f}%')}  max {_num(f'{cl.max():.0f}%')}",
                    f"GPU  avg {_num(f'{np.mean(gl):.0f}%')}  max {_num(f'{gl.max():.0f}%')}",
                ])
            hover_fn = lambda i: f"CPU {cpu_load[i]:.0f}%    GPU {gpu_load[i]:.0f}%"

        elif name == "temp":
            p.setLabel("left", "Temp (°C)")
            _lp(p, t, cpu_temp, pg.mkPen(C["cpu"], width=1), "CPU", get=lambda: cpu_temp)
            _lp(p, t, gpu_temp, pg.mkPen(C["gpu"], width=1), "GPU", get=lambda: gpu_temp)
            y_fn = yf_fn = lambda sl: (min(float(cpu_temp[sl].min()), float(gpu_temp[sl].min())) * 0.9,
                                       max(float(cpu_temp[sl].max()), float(gpu_temp[sl].max())) * 1.1)
            def slice_fn(mask):
                ct, gt = cpu_temp[mask], gpu_temp[mask]
                return "  ".join([
                    f"CPU  avg {_num(f'{np.mean(ct):.0f}°')}  max {_num(f'{ct.max():.0f}°')}",
                    f"GPU  avg {_num(f'{np.mean(gt):.0f}°')}  max {_num(f'{gt.max():.0f}°')}",
                ])
            hover_fn = lambda i: f"CPU {cpu_temp[i]:.0f}°    GPU {gpu_temp[i]:.0f}°"

        elif name == "power":
            p.setLabel("left", "Power (W)")
            if has_cpu_power:
                _lp(p, t, cpu_power, pg.mkPen(C["cpu"], width=1), "CPU", get=lambda: cpu_power)
            _lp(p, t, gpu_power, pg.mkPen(C["gpu"], width=1), "GPU", get=lambda: gpu_power)
            pmax = lambda sl: max(float(gpu_power[sl].max()),
                                  float(cpu_power[sl].max()) if has_cpu_power else 0.0)
            pmin = lambda sl: (min(float(gpu_power[sl].min()), float(cpu_power[sl].min()))
                               if has_cpu_power else float(gpu_power[sl].min()))
            y_fn  = lambda sl: fit_range(pmin(sl), pmax(sl))
            yf_fn = lambda sl: (0, pmax(sl) * Y_PAD_FULL)
            def slice_fn(mask):
                parts = []
                if has_cpu_power:
                    parts.append(f"CPU  avg {_num(f'{np.mean(cpu_power[mask]):.0f}W')}  max {_num(f'{cpu_power[mask].max():.0f}W')}")
                parts.append(f"GPU  avg {_num(f'{np.mean(gpu_power[mask]):.0f}W')}  max {_num(f'{gpu_power[mask].max():.0f}W')}")
                return "  ".join(parts)
            if has_cpu_power:
                hover_fn = lambda i: f"CPU {cpu_power[i]:.0f}W    GPU {gpu_power[i]:.0f}W"
            else:
                hover_fn = lambda i: f"GPU {gpu_power[i]:.0f}W"

        elif name == "memory":
            p.setLabel("left", "Memory (GB)")
            _lp(p, t, ram_used,  pg.mkPen(C["ram"],  width=1), "RAM",  get=lambda: ram_used)
            _lp(p, t, gpu_vram,  pg.mkPen(C["vram"], width=1), "VRAM", get=lambda: gpu_vram)
            _lp(p, t, swap_used, pg.mkPen(C["swap"], width=1), "Swap", get=lambda: swap_used)
            mem_max = lambda sl: max(float(ram_used[sl].max()), float(gpu_vram[sl].max()), float(swap_used[sl].max()))
            mem_min = lambda sl: min(float(ram_used[sl].min()), float(gpu_vram[sl].min()), float(swap_used[sl].min()))
            y_fn  = lambda sl: fit_range(mem_min(sl), mem_max(sl))
            yf_fn = lambda sl: (0, mem_max(sl) * Y_PAD_FULL)
            def slice_fn(mask):
                r, v, s = ram_used[mask], gpu_vram[mask], swap_used[mask]
                return "  ".join([
                    f"RAM  avg {_num(f'{np.mean(r):.1f} GB')}  max {_num(f'{r.max():.1f} GB')}",
                    f"VRAM  avg {_num(f'{np.mean(v):.1f} GB')}  max {_num(f'{v.max():.1f} GB')}",
                    f"Swap  max {_num(f'{s.max():.1f} GB')}",
                ])
            hover_fn = lambda i: (
                f"RAM {ram_used[i]:.1f} GB    VRAM {gpu_vram[i]:.1f} GB    Swap {swap_used[i]:.1f} GB"
            )

        # ── common items attached to this panel ──
        y_default, y_full = y_fn(slice(None)), yf_fn(slice(None))
        # The default (whole-session) stats are just slice_fn over every row.
        default_stats = slice_fn(full_mask)
        vl  = pg.PlotCurveItem([], [], pen=cross_pen)
        p.addItem(vl, ignoreBounds=True)
        st  = make_text_item(p, overlay=True)
        st.setHtml(default_stats)
        hv  = make_text_item(p, anchor=(0, 1), visible=False, overlay=True)
        reg = make_region(p)

        panel_states.append({
            "name": name,
            "plot": p, "vb": vb,
            "vl": vl, "st": st, "hv": hv, "reg": reg, "bw": 0.0,
            "default_stats": default_stats,
            "slice_fn": slice_fn, "hover_fn": hover_fn,
            "y_default": y_default, "y_full": y_full,
            "y_fn": y_fn, "yf_fn": yf_fn,
        })

    # ── span-duration label centered along the top of the last panel, clear of the
    # bottom hover readouts (and of the top-left stats / top-right legend). ──
    last_p   = panel_states[-1]["plot"]
    dur_item = make_text_item(last_p, anchor=(0.5, 0), visible=False, overlay=True)

    # ── stats pinning ──
    # Overlay labels live in PlotItem pixel space so they don't move during
    # pan/zoom. Only need to reposition on layout/resize.
    def pin_stats():
        for ps in panel_states:
            vb = ps["vb"]
            ps["st"].setPos(vb.pos().x() + 4, vb.pos().y() + 4)

    def pin_dur():
        vb = last_p.vb
        dur_item.setPos(round(vb.pos().x() + vb.size().width() / 2), vb.pos().y() + 4)

    first_p.geometryChanged.connect(lambda: (pin_stats(), pin_dur()))
    _keep = []
    _span_active = False   # whether a span selection currently drives the stats

    # ── crosshair + hover ──
    def nearest(x):
        idx = int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))
        if idx > 0 and abs(t[idx - 1] - x) < abs(t[idx] - x):
            idx -= 1
        return idx

    _last_idx = -1
    _mouse_in = True   # cursor inside the widget? (gates stale rate-limited moves)

    def place_overlays(idx, set_text=True):
        xi = float(t[idx])
        # All panels share the x-link, so take the x-range from first_p (the panel
        # whose signal drives repositioning); reading each panel's own viewRange
        # can lag a frame behind during pan as the link propagates.
        vx0, vx1 = first_p.vb.viewRange()[0]
        fracx = (xi - vx0) / (vx1 - vx0) if vx1 > vx0 else 0.0
        for ps2 in panel_states:
            vb2 = ps2["vb"]
            ps2["vl"].setData([xi, xi], vb2.viewRange()[1])
            # Pixel-space readout pinned to the panel bottom (clear of the
            # top-left stats and top-right legend), centered on the cursor
            # but clamped flush to whichever edge it would otherwise cross.
            hv = ps2["hv"]
            if set_text:
                hv.setText(ps2["hover_fn"](idx))
                ps2["bw"] = hv.boundingRect().width()  # width only changes with the text
            bw    = ps2["bw"]
            cx    = vb2.pos().x() + fracx * vb2.size().width()
            left, right = vb2.pos().x(), vb2.pos().x() + vb2.size().width()
            bx = max(left + 2, min(cx - bw / 2, right - bw - 2))
            by = vb2.pos().y() + vb2.size().height() - 3
            hv.setPos(round(bx), round(by))
            hv.setVisible(True)

    def clear_crosshair():
        nonlocal _last_idx
        if _last_idx != -1:
            _last_idx = -1
            for ps in panel_states:
                ps["vl"].setData([], [])
                ps["hv"].setVisible(False)

    def on_mouse_move(scene_pos):
        nonlocal _last_idx
        if not _mouse_in:
            return  # ignore the rate-limited move buffered before the cursor left
        for ps in panel_states:
            if ps["plot"].sceneBoundingRect().contains(scene_pos):
                idx = nearest(ps["plot"].vb.mapSceneToView(scene_pos).x())
                if idx != _last_idx:
                    _last_idx = idx
                    place_overlays(idx)
                return
        clear_crosshair()

    # The mouse-move proxy stops firing once the cursor exits the widget, which
    # would leave the crosshair frozen at the edge; clear it on leave. A move
    # event buffered by the proxy may still arrive after the leave, so gate it on
    # _mouse_in (re-armed on enter) to stop the crosshair flickering back.
    _prev_leave, _prev_enter = win.leaveEvent, win.enterEvent
    def _leave(ev):
        nonlocal _mouse_in
        _mouse_in = False
        clear_crosshair()
        _prev_leave(ev)
    def _enter(ev):
        nonlocal _mouse_in
        _mouse_in = True
        _prev_enter(ev)
    win.leaveEvent = _leave
    win.enterEvent = _enter

    # Keep the pixel-space readouts glued to the crosshair as the view moves: a
    # pan/zoom leaves idx unchanged (so on_mouse_move debounces out) but shifts
    # where that data point lands on screen. Text is unchanged, so skip setText.
    def reposition_overlays():
        if _last_idx != -1:
            place_overlays(_last_idx, set_text=False)

    first_p.sigXRangeChanged.connect(lambda *_: reposition_overlays())

    _keep.append(pg.SignalProxy(
        first_p.scene().sigMouseMoved, rateLimit=30,
        slot=lambda ev: on_mouse_move(ev[0]),
    ))

    # ── span selection ──
    def on_span_drag(lo, hi):
        for ps in panel_states:
            ps["reg"].setRegion([lo, hi])
            ps["reg"].setVisible(True)

    def on_span_finished(lo, hi):
        nonlocal _span_active
        on_span_drag(lo, hi)
        mask = (t >= lo) & (t <= hi)
        if mask.sum() < 2:
            return
        _span_active = True
        for ps in panel_states:
            ps["st"].setHtml(ps["slice_fn"](mask))
        dur_item.setText(fmt_duration(hi - lo))
        dur_item.setVisible(True)

    def on_span_cleared(*_):
        nonlocal _span_active
        _span_active = False
        for ps in panel_states:
            ps["reg"].setVisible(False)
            ps["st"].setHtml(ps["default_stats"])
        dur_item.setVisible(False)

    for ps in panel_states:
        ps["vb"].sigSpanDrag.connect(on_span_drag)
        ps["vb"].sigSpanFinished.connect(on_span_finished)
        ps["vb"].sigSpanCleared.connect(on_span_cleared)

    # ── x limits ──
    avg_dt    = (t_end - t0) / max(1, len(t) - 1)
    min_range = avg_dt * 200
    for ps in panel_states:
        ps["vb"].setLimits(xMin=t0, xMax=t_end, minXRange=min_range)

    # ── tiered LOD: pick the finest pyramid level whose in-window point count
    # stays under the cap, so rendered detail tracks the zoom level. Cheap per
    # frame (two searchsorts); all curves share the x link and the same level
    # structure, so one index drives them all and setData fires only on a change. ──
    n_full = len(t)

    def update_lod():
        (x0, x1), _ = first_p.vb.viewRange()
        lo, hi  = np.searchsorted(t, (x0, x1))
        visible = int(hi - lo)
        frac    = visible / n_full
        levels  = lod_curves[0]["levels"]
        idx     = len(levels) - 1
        for i, (lx, _ly) in enumerate(levels):
            if len(lx) * frac <= LOD_MAX_VISIBLE:
                idx = i
                break
        for c in lod_curves:
            if c["idx"] != idx:
                c["idx"] = idx
                c["item"].setData(*c["levels"][idx], skipFiniteCheck=True)

    if lod_curves:
        first_p.sigXRangeChanged.connect(lambda *_: update_lod())

    # ── Y key: toggle constrained ↔ full Y for fps/frametime only ──
    _full_y = False

    def toggle_y():
        nonlocal _full_y
        _full_y = not _full_y
        for ps in panel_states:
            if ps["name"] in ("fps", "frametime"):
                ps["plot"].setYRange(*(ps["y_full"] if _full_y else ps["y_default"]), padding=0)

    for ps in panel_states:
        ps["plot"].hideButtons()

    # ── initial state ──
    first_p.setXRange(t0, t_end, padding=0)
    for ps in panel_states:
        ps["plot"].setYRange(*ps["y_default"], padding=0)

    legends = [ps["plot"].legend for ps in panel_states]

    def _key_press(ev):
        k = ev.key()
        if k == QtCore.Qt.Key.Key_Y:
            toggle_y()
        elif k == QtCore.Qt.Key.Key_L:
            vis = not legends[0].isVisible()
            for lg in legends:
                lg.setVisible(vis)
        elif k == QtCore.Qt.Key.Key_F and live:
            toggle_follow()
        elif k == QtCore.Qt.Key.Key_Q:
            win.close()
        else:
            type(win).keyPressEvent(win, ev)
    win.keyPressEvent = _key_press

    win.show()
    pin_stats()
    pin_dur()

    # ── live mode: poll the growing log and fold new rows into the view ──
    if live:
        _following   = True   # auto-track the live edge until the user pans away
        _prog_xrange = None   # last x-range we set ourselves (to detect user pans)

        def set_xrange_live():
            nonlocal _prog_xrange
            lo = t0 if full_session else max(t0, t_end - follow_window)
            _prog_xrange = (lo, t_end)
            first_p.setXRange(lo, t_end, padding=0)

        def on_xrange_changed(*_):
            # A pan/zoom that doesn't match our own set means the user moved the
            # view; stop chasing the live edge until they re-enable follow (F).
            nonlocal _following
            if _following and _prog_xrange is not None:
                (x0, x1), _ = first_p.vb.viewRange()
                if abs(x0 - _prog_xrange[0]) > 1e-6 or abs(x1 - _prog_xrange[1]) > 1e-6:
                    _following = False
        first_p.sigXRangeChanged.connect(on_xrange_changed)

        def toggle_follow():
            nonlocal _following
            _following = not _following
            if _following:
                set_xrange_live()

        def fit_y():
            # Fit each y-axis to the data inside the current x-range (the rolling
            # window while following), not the whole session, so an old spike
            # doesn't keep every axis zoomed out.
            (x0, x1), _ = first_p.vb.viewRange()
            lo = int(np.searchsorted(t, x0, "left"))
            hi = int(np.searchsorted(t, x1, "right"))
            sl = slice(lo, max(hi, lo + 1))
            for ps in panel_states:
                ps["y_default"] = ps["y_fn"](sl)
                ps["y_full"]    = ps["yf_fn"](sl)
                full = _full_y and ps["name"] in ("fps", "frametime")
                ps["plot"].setYRange(*(ps["y_full"] if full else ps["y_default"]), padding=0)
        # Refit y on any view change (follow tick or a manual pan/zoom).
        first_p.sigXRangeChanged.connect(lambda *_: fit_y())

        def on_tick():
            nonlocal data
            rows = tail.read_rows()
            if rows is None:
                return
            rows = filter_fps(rows)
            if len(rows) == 0:
                return
            data = np.concatenate([data, rows], axis=0)
            recompute()
            for it, get in live_curves:
                it.setData(t, get(), skipFiniteCheck=True)
            for ln, get in live_lines:
                ln.setValue(get())
            for ps in panel_states:
                ps["default_stats"] = ps["slice_fn"](full_mask)
                if not _span_active:
                    ps["st"].setHtml(ps["default_stats"])
            min_range = (t_end - t0) / max(1, len(t) - 1) * 200
            for ps in panel_states:
                ps["vb"].setLimits(xMin=t0, xMax=t_end, minXRange=min_range)
            bottom = last_p.getAxis("bottom")
            bottom._show_hours = t_end >= 3600
            last_p.setLabel("bottom", f"[{fmt_duration(t_end)}]")
            if _following:
                set_xrange_live()  # shifts the window → sigXRangeChanged → fit_y
            else:
                fit_y()            # window unchanged but its contents may have

        timer = QtCore.QTimer()
        timer.timeout.connect(on_tick)
        timer.start(poll_ms)
        _keep.append(timer)
        set_xrange_live()
        fit_y()

    return win, tuple(_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangoHud plotter (pyqtgraph)")
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--smooth",  type=int,   default=30)
    parser.add_argument("--jitter", type=float, default=1.5,
                        help="Jitter threshold as a multiple of the rolling average (default: 1.5)")
    parser.add_argument("--hitch-ms", type=float, default=100.0, metavar="MS",
                        help="Frametime (ms) at or above which a frame is a hitch (default: 100)")
    parser.add_argument("--stall-ms", type=float, default=200.0, metavar="MS",
                        help="Frametime (ms) at or above which a frame is a stall (default: 200)")
    parser.add_argument("--show",    metavar="PANELS",
                        help=f"Panels to display, comma-separated. Options: {','.join(ALL_PANELS)}")
    parser.add_argument("--except",  dest="hide", metavar="PANELS",
                        help="Panels to hide from default set, comma-separated")
    parser.add_argument("-o", "--output",
                        help="Save screenshot to file instead of displaying")
    parser.add_argument("--max-fps", type=float, default=1000.0, metavar="N",
                        help="Drop frames at or above N FPS as outliers (default: 1000)")
    parser.add_argument("--keep-all", action="store_true",
                        help="Disable outlier filtering; keep all frames regardless of FPS")
    parser.add_argument("--trim-start", type=float, default=0.0, metavar="SEC",
                        help="Drop the first SEC seconds of the session")
    parser.add_argument("--trim-end", type=float, default=0.0, metavar="SEC",
                        help="Drop the last SEC seconds of the session")
    parser.add_argument("--live", action="store_true",
                        help="Update the plot as the log file is written")
    parser.add_argument("--full-session", action="store_true",
                        help="Live: show the whole growing session instead of a rolling window")
    parser.add_argument("--window", type=float, default=60.0, metavar="SEC",
                        help="Live: rolling window length in seconds (default: 60)")
    parser.add_argument("--poll", type=int, default=1000, metavar="MS",
                        help="Live: log poll interval in milliseconds (default: 1000)")
    args = parser.parse_args()

    if args.show:
        show_set = set(args.show.split(","))
    elif args.hide:
        show_set = set(ALL_PANELS) - set(args.hide.split(","))
    else:
        show_set = set(ALL_PANELS)

    if not args.logs:
        csvs = sorted(
            f for f in Path(__file__).parent.glob("*.csv")
            if not f.stem.endswith("_summary")
        )
        if not csvs:
            print("No .csv files found.")
            sys.exit(1)
        args.logs = [str(csvs[-1])]
        print(f"Using {args.logs[0]}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    max_fps = None if args.keep_all else args.max_fps

    if args.output:
        win, _ = plot(args.logs[0], smooth=args.smooth, jitter_threshold=args.jitter,
                      show=show_set, max_fps=max_fps,
                      trim_start=args.trim_start, trim_end=args.trim_end,
                      hitch_ms=args.hitch_ms, stall_ms=args.stall_ms)
        app.processEvents()
        win.grab().save(args.output)
        print(f"Saved to {args.output}")
        sys.exit(0)

    wins = [plot(p, smooth=args.smooth, jitter_threshold=args.jitter, show=show_set,
                 max_fps=max_fps, trim_start=args.trim_start, trim_end=args.trim_end,
                 hitch_ms=args.hitch_ms, stall_ms=args.stall_ms,
                 live=args.live, full_session=args.full_session,
                 follow_window=args.window, poll_ms=args.poll)
            for p in args.logs]
    sys.exit(app.exec())
