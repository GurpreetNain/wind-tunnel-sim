"""
Live interactive wind tunnel -- hover case, live wind.

Separate tool from wind_tunnel_gui.py (untouched) -- this one runs a
CONTINUOUS clamped ball-socket simulation (position frozen, free rotation,
same physics/wind-reprojection as wind_tunnel_gui.py's clamped mode) at
real-time-ish pace, driven live by four sliders:

    - Elevon 1, Elevon 2 (deg, independent channels)
    - Rotor tilt (deg, uniform across all 4 rotors)
    - Wind velocity (m/s) -- re-read every physics step, not just once at
      startup, so you can drag it from 0 up and watch the vehicle's own
      transient response as real airflow builds up over the wing.

Rotor SPEED is fixed at the exact hover trim split (169.00885 / 188.38417
rev/s front/aft) -- that's what makes this "the hover case." Everything
starts at true hover (wind=0, so pitch starts at exactly 0 deg, matching
the perfect-hover result validated elsewhere this session), and Reset
zeros every slider, not just the vehicle's state.

Run:
    python scripts/wind_tunnel_live.py
"""
import os
import sys
from collections import deque

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QSlider, QLabel, QGroupBox, QPushButton, QGridLayout, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.actuators.params import InterceptorParams
from src.models.Interceptor import Interceptor
from src.common.utils import quat2rot

import airframe_view

VEHICLE_CONFIG = os.path.join(parent_dir, "config", "Vehicles", "DeltaV", "DeltaV_Interceptor.json")
DT = 0.002
SUBSTEPS_PER_TICK = 10   # 10 * 0.002s = 0.02s simulated per 20ms timer tick -- real-time pace
TIMER_MS = 20
START_ALTITUDE_M = 5.0
WIND_MPS_INIT = 0.0
WIND_MPS_MAX = 60.0
ROTOR_SPEED_FRONT = 169.00885
ROTOR_SPEED_AFT = 188.38417
HISTORY_SECONDS = 12.0

# Slider resolution: Qt sliders are integer-only, so degrees are stored as (angle * SCALE)
SCALE = 10   # 0.1 deg resolution


class LiveWindTunnel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeltaV Live Wind Tunnel -- Hover Case, Live Sliders")
        self.resize(1400, 850)

        self.params = InterceptorParams.from_json(VEHICLE_CONFIG)
        self.t = 0.0
        self.history_len = int(HISTORY_SECONDS / (SUBSTEPS_PER_TICK * DT))
        self.hist_t = deque(maxlen=self.history_len)
        self.hist_pitch = deque(maxlen=self.history_len)
        self.hist_roll = deque(maxlen=self.history_len)
        self.hist_yaw = deque(maxlen=self.history_len)
        self.hist_alpha_wing = deque(maxlen=self.history_len)

        self._build_plant()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addLayout(self._build_left_panel(), stretch=0)
        root.addLayout(self._build_right_panel(), stretch=1)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(TIMER_MS)

    # ------------------------------------------------------------- plant setup
    def _build_plant(self):
        self.plant = Interceptor(params=self.params, id=1)
        self.plant._state_vector[2] = -START_ALTITUDE_M
        self.plant._state_vector[3] = WIND_MPS_INIT
        self.frozen_position = self.plant._state_vector[0:3].copy()

    def _reset(self):
        self._build_plant()
        self.t = 0.0
        self.hist_t.clear()
        self.hist_pitch.clear()
        self.hist_roll.clear()
        self.hist_yaw.clear()
        self.hist_alpha_wing.clear()
        self.slider_elevon1.setValue(0)
        self.slider_elevon2.setValue(0)
        self.slider_tilt.setValue(0)
        self.slider_wind.setValue(0)

    # ------------------------------------------------------------- left panel
    def _build_left_panel(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Rotor speed (fixed, hover trim): front={ROTOR_SPEED_FRONT:.3f} "
                                 f"aft={ROTOR_SPEED_AFT:.3f} rev/s"))

        self.slider_elevon1, lbl1 = self._make_slider("Elevon 1", -30.0, 30.0, 0.0, "deg")
        self.slider_elevon2, lbl2 = self._make_slider("Elevon 2", -30.0, 30.0, 0.0, "deg")
        self.slider_tilt, lbl3 = self._make_slider("Rotor tilt (all 4)", 0.0, 90.0, 0.0, "deg")
        self.slider_wind, lbl4 = self._make_slider("Wind velocity", 0.0, WIND_MPS_MAX, WIND_MPS_INIT, "m/s")

        # Elevons LINKED by default. Two independent sliders cannot physically be
        # dragged at the same instant, so moving one alone briefly commands a
        # differential (roll) deflection. Roll on this airframe has damping but NO
        # restoring stiffness and there is no controller in the loop, so that brief
        # transient leaves a PERMANENT bank angle that never washes out -- which is
        # what produced the spurious ~90 deg roll. Linked, delta_a stays exactly 0
        # and a symmetric input gives pure pitch.
        self.chk_link = QCheckBox("Link elevons (symmetric = pure pitch, no roll)")
        self.chk_link.setChecked(True)
        self._syncing = False
        self.slider_elevon1.valueChanged.connect(lambda v: self._sync_elevons(self.slider_elevon1))
        self.slider_elevon2.valueChanged.connect(lambda v: self._sync_elevons(self.slider_elevon2))

        for s, l in [(self.slider_elevon1, lbl1), (self.slider_elevon2, lbl2)]:
            layout.addWidget(l)
            layout.addWidget(s)
        layout.addWidget(self.chk_link)
        for s, l in [(self.slider_tilt, lbl3), (self.slider_wind, lbl4)]:
            layout.addWidget(l)
            layout.addWidget(s)

        btn_reset = QPushButton("Reset simulation")
        btn_reset.clicked.connect(self._reset)
        layout.addWidget(btn_reset)

        layout.addWidget(QLabel(""))
        self.readout = QLabel()
        self.readout.setFont(QFont("Consolas", 11))
        layout.addWidget(self.readout)

        layout.addStretch()
        return layout

    def _sync_elevons(self, source):
        """Mirror one elevon slider onto the other while 'Link elevons' is checked."""
        if self._syncing or not self.chk_link.isChecked():
            return
        other = self.slider_elevon2 if source is self.slider_elevon1 else self.slider_elevon1
        self._syncing = True
        other.setValue(source.value())
        self._syncing = False

    def _make_slider(self, label, lo, hi, init, unit="deg"):
        lbl = QLabel(f"{label}: {init:.1f} {unit}")
        s = QSlider(Qt.Orientation.Horizontal)
        s.setMinimum(int(lo * SCALE))
        s.setMaximum(int(hi * SCALE))
        s.setValue(int(init * SCALE))
        s.valueChanged.connect(lambda v, lbl=lbl, label=label, unit=unit: lbl.setText(f"{label}: {v/SCALE:.1f} {unit}"))
        return s, lbl

    # ------------------------------------------------------------ right panel
    def _build_right_panel(self):
        layout = QVBoxLayout()
        top = QHBoxLayout()

        self.fig_3d = Figure(figsize=(4, 4))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        top.addWidget(self._wrap("3D Attitude (live)", self.canvas_3d))

        self.fig_hist = Figure(figsize=(7, 4))
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.ax_att = self.fig_hist.add_subplot(211)
        self.ax_alpha = self.fig_hist.add_subplot(212)
        top.addWidget(self._wrap(f"Last {HISTORY_SECONDS:.0f}s: Attitude & alpha_wing", self.canvas_hist))

        layout.addLayout(top)
        return layout

    def _wrap(self, title, canvas):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        v.addWidget(canvas)
        return box

    # ------------------------------------------------------------------ tick
    def _tick(self):
        elevon1 = np.radians(self.slider_elevon1.value() / SCALE)
        elevon2 = np.radians(self.slider_elevon2.value() / SCALE)
        tilt = np.radians(self.slider_tilt.value() / SCALE)
        control_vector = np.array([elevon1, elevon2, tilt, tilt, tilt, tilt,
                                    ROTOR_SPEED_FRONT, ROTOR_SPEED_AFT, ROTOR_SPEED_FRONT, ROTOR_SPEED_AFT])
        self.plant._control_vector = control_vector

        wind_mps = self.slider_wind.value() / SCALE
        world_wind_vector = np.array([wind_mps, 0.0, 0.0])

        for _ in range(SUBSTEPS_PER_TICK):
            self.plant.state_update(DT)
            # Ball-socket clamp: position frozen, wind fixed in the world frame (read
            # live from the slider every substep), re-projected into the (rotating)
            # body frame -- same physics as wind_tunnel_gui.py's clamped mode.
            self.plant._state_vector[0:3] = self.frozen_position
            R_world_from_body = quat2rot(self.plant._state_vector[6:10])
            self.plant._state_vector[3:6] = R_world_from_body.T @ world_wind_vector
            self.t += DT

        sv = self.plant.get_state_vector()
        if not np.all(np.isfinite(sv)):
            self.readout.setText("DIVERGED (NaN/Inf) -- click Reset simulation.")
            self.timer.stop()
            return

        d = self.plant.last_diagnostics
        q = sv[6:10]
        roll, pitch, yaw = self._roll(q), self._pitch(q), self._yaw(q)
        alpha_wing_deg = np.degrees(d['alpha_wing'])

        self.hist_t.append(self.t)
        self.hist_pitch.append(pitch)
        self.hist_roll.append(roll)
        self.hist_yaw.append(yaw)
        self.hist_alpha_wing.append(alpha_wing_deg)

        delta_e_deg = 0.5 * (self.slider_elevon1.value() + self.slider_elevon2.value()) / SCALE
        delta_a_deg = 0.5 * (self.slider_elevon2.value() - self.slider_elevon1.value()) / SCALE

        self.readout.setText(
            f"t = {self.t:6.2f} s\n"
            f"Wind:  {wind_mps:6.1f} m/s\n"
            f"delta_e (pitch): {delta_e_deg:6.1f} deg\n"
            f"delta_a (roll):  {delta_a_deg:6.1f} deg\n\n"
            f"Roll:  {roll:8.3f} deg\n"
            f"Pitch: {pitch:8.3f} deg\n"
            f"Yaw:   {yaw:8.3f} deg\n\n"
            f"alpha_wing: {alpha_wing_deg:8.3f} deg\n"
            f"CL: {d['CL']:8.4f}   CD: {d['CD']:8.5f}\n"
            f"Lift: {d['Lclean']:8.3f} N\n"
            f"Drag: {d['D']:8.3f} N\n\n"
            f"Mx: {d['total_moment'][0]:8.4f} N*m\n"
            f"My: {d['total_moment'][1]:8.4f} N*m\n"
            f"Mz: {d['total_moment'][2]:8.4f} N*m"
        )

        self._redraw_3d(q, tilt)
        self._redraw_history()

    @staticmethod
    def _roll(q):
        w, x, y, z = q
        return np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)))

    @staticmethod
    def _pitch(q):
        w, x, y, z = q
        return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))

    @staticmethod
    def _yaw(q):
        w, x, y, z = q
        return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))

    def _redraw_3d(self, q, tilt):
        airframe_view.draw_airframe(self.ax_3d, q, self.params, quat2rot, tilt=tilt)
        self.canvas_3d.draw()

    def _redraw_history(self):
        self.ax_att.clear()
        self.ax_att.plot(self.hist_t, self.hist_roll, label='roll')
        self.ax_att.plot(self.hist_t, self.hist_pitch, label='pitch')
        self.ax_att.plot(self.hist_t, self.hist_yaw, label='yaw')
        self.ax_att.set_ylabel("deg")
        self.ax_att.legend(loc='upper left', fontsize=8)
        self.ax_att.grid(True)
        # Floor the y-range so floating-point zero (~1e-5 deg) isn't autoscaled up
        # into a curve that looks like real motion.
        self._floor_ylim(self.ax_att, 2.0)

        self.ax_alpha.clear()
        self.ax_alpha.plot(self.hist_t, self.hist_alpha_wing, color='tab:red')
        self.ax_alpha.axhline(13.0, color='gray', linestyle='--', linewidth=1)
        self.ax_alpha.axhline(-13.0, color='gray', linestyle='--', linewidth=1)
        self.ax_alpha.set_ylabel("alpha_wing (deg)")
        self.ax_alpha.set_xlabel("t (s)")
        self.ax_alpha.grid(True)

        self.canvas_hist.draw()

    @staticmethod
    def _floor_ylim(ax, min_span_deg):
        """Keep the y-axis at least min_span_deg wide, centred on the current data."""
        lo, hi = ax.get_ylim()
        if hi - lo < min_span_deg:
            mid = 0.5 * (lo + hi)
            ax.set_ylim(mid - min_span_deg / 2.0, mid + min_span_deg / 2.0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveWindTunnel()
    window.show()
    sys.exit(app.exec())
