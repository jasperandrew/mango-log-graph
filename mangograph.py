#!/usr/bin/env python3
"""MangoHud log plotter — pyqtgraph."""

import sys
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
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="same")


def load_log(path):
    with open(path) as f:
        meta    = dict(zip(f.readline().strip().split(","), f.readline().strip().split(",")))
        headers = f.readline().strip().split(",")
        data    = np.loadtxt(f, delimiter=",")
    return meta, data, headers


# LOD pyramid: cap on how many source points a curve renders within the viewport,
# and the decimation step between successive levels. update_lod() picks the finest
# level whose in-window count stays under the cap, so the rendered detail stays in
# the (cap/RATIO, cap] band at every zoom — always crisp, never re-scanning the
# whole series. The cap comfortably exceeds the pixel width, and pyqtgraph's own
# autoDownsample reduces the rest to viewport resolution.
LOD_MAX_VISIBLE = 16_000
LOD_RATIO       = 4


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

def plot(path, smooth=30, stutter_threshold=1.5, show=None, max_fps=1000.0,
         trim_start=0.0, trim_end=0.0):
    meta, data, headers = load_log(path)
    col = {name: i for i, name in enumerate(headers)}

    if max_fps is not None:
        total   = len(data)
        keep    = data[:, col["fps"]] < max_fps
        dropped = total - int(keep.sum())
        if dropped:
            print(f"Dropped {dropped} of {total} rows "
                  f"({dropped / total * 100:.2f}%) at >= {max_fps:g} FPS")
        data = data[keep]

    if trim_start or trim_end:
        rel  = (data[:, col["elapsed"]] - data[0, col["elapsed"]]) / 1e9
        keep = (rel >= trim_start) & (rel <= rel[-1] - trim_end)
        if keep.sum() < 2:
            print(f"Trim removes all but {int(keep.sum())} of {len(data)} rows "
                  f"(session is {fmt_duration(rel[-1])}); skipping trim")
        else:
            print(f"Trimmed {len(data) - int(keep.sum())} rows "
                  f"(-{trim_start:g}s start, -{trim_end:g}s end)")
            data = data[keep]

    t         = (data[:, col["elapsed"]] - data[0, col["elapsed"]]) / 1e9
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

    fps_avg       = rolling_avg(fps, smooth)
    ft_roll       = rolling_avg(ft_ms, smooth)
    has_cpu_power = bool(cpu_power.any())

    fps_mean = float(np.mean(fps))
    fps_p1   = float(np.percentile(fps, 1))
    fps_p01  = float(np.percentile(fps, 0.1))
    ft_mean  = float(np.mean(ft_ms))
    ft_p99   = float(np.percentile(ft_ms, 99))
    sm_count = int((ft_ms > stutter_threshold * ft_roll).sum())
    dur_min  = float(t[-1]) / 60
    sm_rate  = sm_count / dur_min if dur_min > 0 else 0.0
    pacing   = float(np.percentile(np.abs(np.diff(ft_ms)), 99))

    t0, t_end = float(t[0]), float(t[-1])

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
    _lp = lambda *a, **k: lod_plot(*a, registry=lod_curves, **k)

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
            _lp(p, t, fps,     pg.mkPen(C["fps_raw"], width=0.8), "raw")
            _lp(p, t, fps_avg, pg.mkPen(C["fps_avg"], width=1.5), f"{smooth}-frame avg")
            p.addLine(y=fps_p1,   pen=pg.mkPen(C["low_line"], width=1, style=dash))
            p.addLine(y=fps_mean, pen=pg.mkPen(C["pct_line"], width=1, style=dash))
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["low_line"], width=1, style=dash)), "1% low")
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["pct_line"], width=1, style=dash)), "avg")
            y_default = (0, float(fps_avg.max()) * 1.2)
            y_full    = (0, float(fps.max()) * 1.05)
            default_stats = "  ".join([
                f"avg {fps_mean:.0f}", f"1% low {fps_p1:.0f}",
                f"0.1% low {fps_p01:.0f}", f"max {fps.max():.0f}",
            ])
            def slice_fn(mask):
                s = fps[mask]
                return "  ".join([
                    f"avg {np.mean(s):.0f}", f"1% low {np.percentile(s,1):.0f}",
                    f"0.1% low {np.percentile(s,0.1):.0f}", f"max {s.max():.0f}",
                ])
            hover_fn = lambda i: f"t {fmt_duration(float(t[i]))}    FPS {fps[i]:.0f}"

        elif name == "frametime":
            p.setLabel("left", "Frametime (ms)")
            _lp(p, t, ft_ms,   pg.mkPen(C["fps_raw"], width=0.8), "raw")
            _lp(p, t, ft_roll, pg.mkPen(C["ft_avg"],  width=1.5), f"{smooth}-frame avg")
            p.addLine(y=ft_mean, pen=pg.mkPen(C["pct_line"], width=1, style=dash))
            p.addLine(y=ft_p99,  pen=pg.mkPen(C["low_line"], width=1, style=dash))
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["pct_line"], width=1, style=dash)), "avg")
            lg.addItem(pg.PlotDataItem(pen=pg.mkPen(C["low_line"], width=1, style=dash)), "99th pct")
            y_default = (0, float(ft_roll.max()) * 1.2)
            y_full    = (0, float(ft_ms.max()) * 1.05)
            default_stats = "  ".join([
                f"avg {ft_mean:.1f} ms", f"99th {ft_p99:.1f} ms",
                f"stutters {sm_count} ({sm_rate:.1f}/min)",
                f"pacing p99 {pacing:.1f} ms",
            ])
            def slice_fn(mask):
                ft_s = ft_ms[mask]
                t_s  = t[mask]
                # reuse precomputed rolling avg rather than re-convolving the slice
                sc   = int((ft_s > stutter_threshold * ft_roll[mask]).sum())
                dm   = float(t_s[-1] - t_s[0]) / 60
                sr   = sc / dm if dm > 0 else 0.0
                pp99 = float(np.percentile(np.abs(np.diff(ft_s)), 99)) if len(ft_s) > 1 else 0.0
                return "  ".join([
                    f"avg {np.mean(ft_s):.1f} ms",
                    f"99th {np.percentile(ft_s, 99):.1f} ms",
                    f"stutters {sc} ({sr:.1f}/min)",
                    f"pacing p99 {pp99:.1f} ms",
                ])
            hover_fn = lambda i: f"{ft_ms[i]:.2f} ms  (avg {ft_roll[i]:.2f})"

        elif name == "load":
            p.setLabel("left", "Load (%)")
            _lp(p, t, cpu_load, pg.mkPen(C["cpu"], width=1), "CPU")
            _lp(p, t, gpu_load, pg.mkPen(C["gpu"], width=1), "GPU")
            y_default = y_full = (0, 105)
            default_stats = "  ".join([
                f"CPU  avg {np.mean(cpu_load):.0f}%  max {cpu_load.max():.0f}%",
                f"GPU  avg {np.mean(gpu_load):.0f}%  max {gpu_load.max():.0f}%",
            ])
            def slice_fn(mask):
                cl, gl = cpu_load[mask], gpu_load[mask]
                return "  ".join([
                    f"CPU  avg {np.mean(cl):.0f}%  max {cl.max():.0f}%",
                    f"GPU  avg {np.mean(gl):.0f}%  max {gl.max():.0f}%",
                ])
            hover_fn = lambda i: f"CPU {cpu_load[i]:.0f}%    GPU {gpu_load[i]:.0f}%"

        elif name == "temp":
            p.setLabel("left", "Temp (°C)")
            _lp(p, t, cpu_temp, pg.mkPen(C["cpu"], width=1), "CPU")
            _lp(p, t, gpu_temp, pg.mkPen(C["gpu"], width=1), "GPU")
            tmin = min(float(cpu_temp.min()), float(gpu_temp.min())) * 0.9
            tmax = max(float(cpu_temp.max()), float(gpu_temp.max())) * 1.1
            y_default = y_full = (tmin, tmax)
            default_stats = "  ".join([
                f"CPU  avg {np.mean(cpu_temp):.0f}°  max {cpu_temp.max():.0f}°",
                f"GPU  avg {np.mean(gpu_temp):.0f}°  max {gpu_temp.max():.0f}°",
            ])
            def slice_fn(mask):
                ct, gt = cpu_temp[mask], gpu_temp[mask]
                return "  ".join([
                    f"CPU  avg {np.mean(ct):.0f}°  max {ct.max():.0f}°",
                    f"GPU  avg {np.mean(gt):.0f}°  max {gt.max():.0f}°",
                ])
            hover_fn = lambda i: f"CPU {cpu_temp[i]:.0f}°    GPU {gpu_temp[i]:.0f}°"

        elif name == "power":
            p.setLabel("left", "Power (W)")
            if has_cpu_power:
                _lp(p, t, cpu_power, pg.mkPen(C["cpu"], width=1), "CPU")
            _lp(p, t, gpu_power, pg.mkPen(C["gpu"], width=1), "GPU")
            pmax = max(float(gpu_power.max()),
                       float(cpu_power.max()) if has_cpu_power else 0.0)
            y_default = (0, pmax * 1.2)
            y_full    = (0, pmax * 1.05)
            pw_parts  = []
            if has_cpu_power:
                pw_parts.append(f"CPU  avg {np.mean(cpu_power):.0f}W  max {cpu_power.max():.0f}W")
            pw_parts.append(f"GPU  avg {np.mean(gpu_power):.0f}W  max {gpu_power.max():.0f}W")
            default_stats = "  ".join(pw_parts)
            def slice_fn(mask):
                parts = []
                if has_cpu_power:
                    parts.append(f"CPU  avg {np.mean(cpu_power[mask]):.0f}W  max {cpu_power[mask].max():.0f}W")
                parts.append(f"GPU  avg {np.mean(gpu_power[mask]):.0f}W  max {gpu_power[mask].max():.0f}W")
                return "  ".join(parts)
            if has_cpu_power:
                hover_fn = lambda i: f"CPU {cpu_power[i]:.0f}W    GPU {gpu_power[i]:.0f}W"
            else:
                hover_fn = lambda i: f"GPU {gpu_power[i]:.0f}W"

        elif name == "memory":
            p.setLabel("left", "Memory (GB)")
            _lp(p, t, ram_used,  pg.mkPen(C["ram"],  width=1), "RAM")
            _lp(p, t, gpu_vram,  pg.mkPen(C["vram"], width=1), "VRAM")
            _lp(p, t, swap_used, pg.mkPen(C["swap"], width=1), "Swap")
            mem_max = max(float(ram_used.max()), float(gpu_vram.max()), float(swap_used.max()))
            y_default = (0, mem_max * 1.2)
            y_full    = (0, mem_max * 1.05)
            default_stats = "  ".join([
                f"RAM  avg {np.mean(ram_used):.1f}  max {ram_used.max():.1f} GB",
                f"VRAM  avg {np.mean(gpu_vram):.1f}  max {gpu_vram.max():.1f} GB",
                f"Swap  max {swap_used.max():.1f} GB",
            ])
            def slice_fn(mask):
                r, v, s = ram_used[mask], gpu_vram[mask], swap_used[mask]
                return "  ".join([
                    f"RAM  avg {np.mean(r):.1f}  max {r.max():.1f} GB",
                    f"VRAM  avg {np.mean(v):.1f}  max {v.max():.1f} GB",
                    f"Swap  max {s.max():.1f} GB",
                ])
            hover_fn = lambda i: (
                f"RAM {ram_used[i]:.1f} GB    VRAM {gpu_vram[i]:.1f} GB    Swap {swap_used[i]:.1f} GB"
            )

        # ── common items attached to this panel ──
        vl  = pg.PlotCurveItem([], [], pen=cross_pen)
        p.addItem(vl, ignoreBounds=True)
        st  = make_text_item(p, overlay=True)
        st.setText(default_stats)
        hv  = make_text_item(p, anchor=(0, 1), visible=False, overlay=True)
        reg = make_region(p)

        panel_states.append({
            "name": name,
            "plot": p, "vb": vb,
            "vl": vl, "st": st, "hv": hv, "reg": reg,
            "default_stats": default_stats,
            "slice_fn": slice_fn, "hover_fn": hover_fn,
            "y_default": y_default, "y_full": y_full,
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

    # ── crosshair + hover ──
    def nearest(x):
        idx = int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))
        if idx > 0 and abs(t[idx - 1] - x) < abs(t[idx] - x):
            idx -= 1
        return idx

    _last_idx = [-1]

    def on_mouse_move(scene_pos):
        for ps in panel_states:
            if ps["plot"].sceneBoundingRect().contains(scene_pos):
                x   = ps["plot"].vb.mapSceneToView(scene_pos).x()
                idx = nearest(x)
                if idx == _last_idx[0]:
                    return
                _last_idx[0] = idx
                xi  = float(t[idx])
                for ps2 in panel_states:
                    vb2 = ps2["vb"]
                    (vx0, vx1), vy = vb2.viewRange()
                    ps2["vl"].setData([xi, xi], vy)
                    # Pixel-space readout pinned to the panel bottom (clear of the
                    # top-left stats and top-right legend), centered on the cursor
                    # but clamped flush to whichever edge it would otherwise cross.
                    hv = ps2["hv"]
                    hv.setText(ps2["hover_fn"](idx))
                    bw    = hv.boundingRect().width()
                    fracx = (xi - vx0) / (vx1 - vx0) if vx1 > vx0 else 0.0
                    cx    = vb2.pos().x() + fracx * vb2.size().width()
                    left, right = vb2.pos().x(), vb2.pos().x() + vb2.size().width()
                    bx = max(left + 2, min(cx - bw / 2, right - bw - 2))
                    by = vb2.pos().y() + vb2.size().height() - 3
                    hv.setPos(round(bx), round(by))
                    hv.setVisible(True)
                return
        if _last_idx[0] != -1:
            _last_idx[0] = -1
            for ps in panel_states:
                ps["vl"].setData([], [])
                ps["hv"].setVisible(False)

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
        on_span_drag(lo, hi)
        mask = (t >= lo) & (t <= hi)
        if mask.sum() < 2:
            return
        for ps in panel_states:
            ps["st"].setText(ps["slice_fn"](mask))
        dur_item.setText(fmt_duration(hi - lo))
        dur_item.setVisible(True)

    def on_span_cleared(*_):
        for ps in panel_states:
            ps["reg"].setVisible(False)
            ps["st"].setText(ps["default_stats"])
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
        visible = int(np.searchsorted(t, x1) - np.searchsorted(t, x0))
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
        elif k == QtCore.Qt.Key.Key_Q:
            win.close()
        else:
            type(win).keyPressEvent(win, ev)
    win.keyPressEvent = _key_press

    win.show()
    pin_stats()
    pin_dur()

    return win, tuple(_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangoHud plotter (pyqtgraph)")
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--smooth",  type=int,   default=30)
    parser.add_argument("--stutter", type=float, default=1.5)
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
        win, _ = plot(args.logs[0], smooth=args.smooth, stutter_threshold=args.stutter,
                      show=show_set, max_fps=max_fps,
                      trim_start=args.trim_start, trim_end=args.trim_end)
        app.processEvents()
        win.grab().save(args.output)
        print(f"Saved to {args.output}")
        sys.exit(0)

    wins = [plot(p, smooth=args.smooth, stutter_threshold=args.stutter, show=show_set,
                 max_fps=max_fps, trim_start=args.trim_start, trim_end=args.trim_end)
            for p in args.logs]
    sys.exit(app.exec())
