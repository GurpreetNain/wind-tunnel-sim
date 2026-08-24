"""
Flight-path animation for the solved cases -- watch the maneuver happen.

The wind tunnel tools clamp the vehicle's POSITION (ball socket) so they can
read attitude and forces cleanly. That deliberately throws the flight path
away. This tool puts it back: it takes the trim wind_tunnel_solver computed
for a case and flies the vehicle along the trajectory that trim is the
equilibrium FOR, at the solved attitude and actuator state.

WHAT THIS IS, precisely -- worth being straight about:

    The trajectory is KINEMATIC, derived in closed form from the maneuver
    definition, not integrated from the forces. For a level coordinated turn
    the geometry is fully determined by the load factor:

        bank  phi   = arccos(1 / n_load)
        rate  omega = g * tan(phi) / V          (standard turn rate)
        radius R    = V / omega
        pitch theta = solved body alpha         (level turn, so gamma = 0)
        yaw   psi   = omega * t                 (heading follows the tangent)

    and the position is the exact integral of that heading. Every attitude
    angle and every actuator number on screen comes from the real solve --
    only the position is closed-form rather than integrated.

    It is NOT free flight. There is no controller in this codebase, so an
    open-loop integration of these actuator values drifts out of the
    condition within about a second, which is precisely why the testbench
    clamps position in the first place. What this shows is the maneuver the
    solved equilibrium corresponds to.

Run:
    python scripts/maneuver_animation.py
"""
import os
import sys

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QComboBox, QLabel, QGroupBox, QSlider, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.actuators.params import InterceptorParams
from src.common.utils import quat2rot

import wind_tunnel_cases as cases
import wind_tunnel_solver as solver
import airframe_view

VEHICLE_CONFIG = os.path.join(parent_dir, "config", "Vehicles", "DeltaV", "DeltaV_Interceptor.json")
TIMER_MS = 33                 # ~30 fps
MODEL_SCALE = 45.0            # exaggerate the 0.6 m airframe so it reads on a 50 m circle
TRAIL_POINTS = 420


def euler_to_quat(roll, pitch, yaw):
    """3-2-1 (yaw, then pitch, then roll) -> quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


class ManeuverPath:
    """Closed-form trajectory + attitude for one solved case."""

    def __init__(self, params, case_name, case):
        self.name = case_name
        self.p = params
        g = params.g
        self.hover = bool(case.get("hover"))
        self.result = solver.solve_case(params, case)
        self.report = self.result["report"]
        self.tilt = np.radians(np.array([self.result["tilt_FR"], self.result["tilt_AL"],
                                          self.result["tilt_FL"], self.result["tilt_AR"]]))
        self.elevon_deg = self.result["elevon1"]
        self.n = np.array([self.result["n_FR"], self.result["n_AL"],
                           self.result["n_FL"], self.result["n_AR"]])
        self.V = self.result["wind"]

        self.alpha = self._report_value("body alpha=")
        if self.alpha is None:
            self.alpha = self._report_value("alpha=") or 0.0
        self.alpha_wing = self._report_value("alpha_wing held at ")
        if self.alpha_wing is None:
            self.alpha_wing = self._report_value("alpha_wing (post-downwash)=") or 0.0

        if self.hover:
            self.kind = "hover"
            self.n_load, self.bank, self.omega, self.R, self.gamma = 1.0, 0.0, 0.0, 0.0, 0.0
            self.period = 8.0
        else:
            self.n_load = case["n_load"]
            self.gamma = np.radians(case["gamma_deg"])
            if self.n_load > solver.N_LOAD_TURN_THRESHOLD:
                self.kind = "turn"
                self.bank = np.arccos(1.0 / self.n_load)
                self.omega = g * np.tan(self.bank) / self.V     # standard turn rate, rad/s
                self.R = self.V / self.omega
                self.period = 2.0 * np.pi / self.omega
            else:
                self.kind = "climb"
                self.bank, self.omega, self.R = 0.0, 0.0, 0.0
                self.period = 12.0

    def _report_value(self, key):
        """Pull a degrees value out of the solver's own report text."""
        for line in self.report:
            if key in line:
                try:
                    tok = line.split(key)[1].split()[0]
                    return np.radians(float(tok.rstrip("deg,")))
                except (ValueError, IndexError):
                    return None
        return None

    def at(self, t):
        """-> position (NED, z down), quaternion, extras dict."""
        if self.kind == "hover":
            return np.zeros(3), euler_to_quat(0.0, 0.0, 0.0), dict(psi=0.0)
        if self.kind == "climb":
            d = self.V * t
            pos = np.array([d * np.cos(self.gamma), 0.0, -d * np.sin(self.gamma)])
            # body pitch = flight-path angle + incidence
            return pos, euler_to_quat(0.0, self.gamma + self.alpha, 0.0), dict(psi=0.0)
        # level coordinated turn: heading sweeps at omega, position is its integral
        psi = self.omega * t
        pos = np.array([self.R * np.sin(psi), self.R * (1.0 - np.cos(psi)), 0.0])
        return pos, euler_to_quat(self.bank, self.alpha, psi), dict(psi=psi)

    def extent(self):
        if self.kind == "turn":
            return 1.35 * self.R
        if self.kind == "climb":
            return self.V * self.period * 0.62
        return 6.0


class ManeuverWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeltaV Maneuver Animation")
        self.resize(1240, 820)
        self.params = InterceptorParams.from_json(VEHICLE_CONFIG)
        self.t = 0.0
        self.playing = True
        self.path = None
        self.trail = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addLayout(self._build_left(), stretch=0)
        root.addLayout(self._build_right(), stretch=1)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(TIMER_MS)
        self._load_case()

    # ------------------------------------------------------------ controls
    def _build_left(self):
        col = QVBoxLayout()
        col.addWidget(QLabel("Maneuver:"))
        self.combo = QComboBox()
        self.combo.addItems([k for k, v in cases.CASES.items() if not v.get("interactive")])
        self.combo.currentIndexChanged.connect(self._load_case)
        col.addWidget(self.combo)

        row = QHBoxLayout()
        self.btn_play = QPushButton("Pause")
        self.btn_play.clicked.connect(self._toggle)
        row.addWidget(self.btn_play)
        btn_r = QPushButton("Restart")
        btn_r.clicked.connect(self._restart)
        row.addWidget(btn_r)
        col.addLayout(row)

        col.addWidget(QLabel("Playback speed"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(200)
        self.slider.setValue(35)
        col.addWidget(self.slider)

        self.chk_trail = QCheckBox("Show flight path")
        self.chk_trail.setChecked(True)
        col.addWidget(self.chk_trail)
        self.chk_follow = QCheckBox("Chase camera")
        self.chk_follow.setChecked(False)
        col.addWidget(self.chk_follow)

        self.readout = QLabel()
        self.readout.setFont(QFont("Consolas", 10))
        self.readout.setAlignment(Qt.AlignmentFlag.AlignTop)
        col.addWidget(self.readout)
        col.addStretch()
        return col

    def _build_right(self):
        col = QVBoxLayout()
        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        box = QGroupBox("Flight path (kinematic, from the solved trim)")
        v = QVBoxLayout(box)
        v.addWidget(self.canvas)
        col.addWidget(box)
        return col

    # ------------------------------------------------------------ actions
    def _toggle(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def _restart(self):
        self.t = 0.0
        self.trail = []

    def _load_case(self):
        name = self.combo.currentText()
        if not name:
            return
        self.path = ManeuverPath(self.params, name, cases.CASES[name])
        self.t = 0.0
        self.trail = []

    # ------------------------------------------------------------ frame
    def _tick(self):
        if self.path is None:
            return
        if self.playing:
            self.t += (self.slider.value() / 100.0) * (TIMER_MS / 1000.0)
            if self.t > self.path.period:
                self.t -= self.path.period
                self.trail = []
        pos, quat, extra = self.path.at(self.t)
        self.trail.append(pos.copy())
        if len(self.trail) > TRAIL_POINTS:
            self.trail.pop(0)
        self._draw(pos, quat)
        self._readout(pos, extra)

    def _draw(self, pos, quat):
        p = self.path
        ax = self.ax
        ax.clear()

        if self.chk_trail.isChecked():
            if p.kind == "turn":
                psi = np.linspace(0.0, 2.0 * np.pi, 260)
                ax.plot(p.R * np.sin(psi), p.R * (1.0 - np.cos(psi)), np.zeros_like(psi),
                        color="#9aa6ad", linewidth=1.0, linestyle="--")
            if len(self.trail) > 1:
                tr = np.array(self.trail)
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#1d5d87", linewidth=2.0)

        airframe_view.draw_airframe(ax, quat, self.params, quat2rot,
                                    tilt=p.tilt, center=pos, scale=MODEL_SCALE,
                                    clear=False, set_limits=False)

        if self.chk_follow.isChecked():
            e = max(p.extent() * 0.22, 12.0)
            cx, cy, cz = pos
        else:
            e = p.extent()
            cx = cy = cz = 0.0
            if p.kind == "turn":
                cy = p.R
            elif p.kind == "climb":
                cx = e * 0.45
                cz = -e * 0.45 * np.tan(p.gamma)
        ax.set_xlim(cx - e, cx + e)
        ax.set_ylim(cy - e, cy + e)
        ax.set_zlim(cz - e, cz + e)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.invert_zaxis()
        ax.set_xlabel("North (m)", fontsize=8)
        ax.set_ylabel("East (m)", fontsize=8)
        ax.set_zlabel("Up (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True)
        self.canvas.draw()

    def _readout(self, pos, extra):
        p = self.path
        lines = [f"t = {self.t:6.2f} s  of {p.period:.2f} s", ""]
        if p.kind == "turn":
            lines += [f"load factor  {p.n_load:6.2f} g",
                      f"bank         {np.degrees(p.bank):6.2f} deg",
                      f"radius       {p.R:6.2f} m",
                      f"turn rate    {np.degrees(p.omega):6.2f} deg/s",
                      f"lap time     {p.period:6.2f} s",
                      f"heading      {np.degrees(extra['psi']) % 360.0:6.1f} deg"]
        elif p.kind == "climb":
            lines += [f"flight path  {np.degrees(p.gamma):6.3f} deg",
                      f"climb rate   {p.V * np.sin(p.gamma):6.2f} m/s",
                      f"distance     {p.V * self.t:6.1f} m"]
        else:
            lines += ["stationary hover"]
        lines += ["",
                  f"airspeed     {p.V:6.2f} m/s",
                  f"body pitch   {np.degrees(p.alpha):6.2f} deg",
                  f"alpha_wing   {np.degrees(p.alpha_wing):6.2f} deg",
                  f"  stall cap  {np.degrees(self.params.alpha_stall):6.2f} deg",
                  "",
                  f"rotor tilt   {np.degrees(p.tilt[0]):6.2f} / {np.degrees(p.tilt[1]):6.2f} deg",
                  f"n front/aft  {p.n[0]:6.1f} / {p.n[1]:6.1f} rev/s",
                  f"             {p.n[0] * 60:6.0f} / {p.n[1] * 60:6.0f} RPM",
                  f"elevons      {p.elevon_deg:6.2f} deg",
                  "",
                  f"altitude     {-pos[2]:6.2f} m",
                  f"position     {pos[0]:7.1f}, {pos[1]:7.1f} m"]
        self.readout.setText("\n".join(lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ManeuverWindow()
    w.show()
    sys.exit(app.exec())
