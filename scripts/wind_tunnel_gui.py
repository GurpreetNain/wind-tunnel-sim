"""
Wind tunnel GUI for the DeltaV Interceptor.

Three ways to drive it, all ending in the same clamped/free-flight run:
  - "Solve Case & Run": pick a named flight condition (wind_tunnel_cases.py
    -- hover, head-on interception, coordinated turn, ...) and it solves for
    every actuator value that condition needs (wind_tunnel_solver.py),
    writes them into wind_tunnel_input.py, and runs. Nothing manual.
  - "Solve Trim (manual RPM) & Run": you set rotor SPEED and wind velocity
    in wind_tunnel_input.py yourself; it solves for the tilt/elevon that
    balance THAT speed, writes them back, and runs.
  - "Run (file as-is)": runs exactly whatever is currently written in
    wind_tunnel_input.py, no solving.

The real rigid-body physics (src/models/Interceptor.py) is driven forward
in time with the actuator command held constant -- no controller, no
allocator beyond what a case solve computes, nothing else -- and the result
is shown as an animated 3D attitude view, a results-text panel, and line
graphs (CL vs CD, CL vs alpha_wing, Lift, Drag, Moments, Roll/Pitch/Yaw).

Run:
    python scripts/wind_tunnel_gui.py
"""
import os
import sys
import importlib

import numpy as np
from scipy.optimize import least_squares

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QSlider, QGroupBox, QTextEdit, QComboBox, QLabel)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.actuators.params import InterceptorParams
from src.actuators import rotors, lifting_body, elevons
from src.models.Interceptor import Interceptor
from src.common.utils import quat2rot

import wind_tunnel_input as cfg
import wind_tunnel_cases as cases
import wind_tunnel_solver as solver
import airframe_view

VEHICLE_CONFIG = os.path.join(parent_dir, "config", "Vehicles", "DeltaV", "DeltaV_Interceptor.json")
DT = 0.002
START_ALTITUDE_M = 5.0   # negative-Z is up in NED; start off the ground so the ground clamp doesn't interfere


class WindTunnelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeltaV Wind Tunnel")
        self.resize(1500, 950)

        self.params = InterceptorParams.from_json(VEHICLE_CONFIG)
        self.log = None   # populated by run()
        self.frame = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_control_bar())
        root.addLayout(self._build_plot_grid())

        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------ control bar
    def _build_control_bar(self):
        layout = QHBoxLayout()

        case_col = QVBoxLayout()
        case_col.addWidget(QLabel("Named case:"))
        self.combo_case = QComboBox()
        self.combo_case.addItems(list(cases.CASES.keys()))
        case_col.addWidget(self.combo_case)
        self.btn_case = QPushButton("Solve Case && Run")
        self.btn_case.clicked.connect(self._on_solve_case)
        case_col.addWidget(self.btn_case)
        layout.addLayout(case_col)

        btn_col = QVBoxLayout()
        self.btn_solve = QPushButton("Solve Trim (manual RPM) && Run")
        self.btn_solve.clicked.connect(self._on_solve_trim)
        btn_col.addWidget(self.btn_solve)

        self.btn_run = QPushButton("Run (file as-is)")
        self.btn_run.clicked.connect(self._on_run)
        btn_col.addWidget(self.btn_run)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_play.setEnabled(False)
        btn_col.addWidget(self.btn_play)
        layout.addLayout(btn_col)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, stretch=1)

        return layout

    # ------------------------------------------------------------- plot grid
    def _build_plot_grid(self):
        layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row3 = QHBoxLayout()

        self.fig_3d = Figure(figsize=(5, 5))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        row1.addWidget(self._wrap("3D Attitude", self.canvas_3d))

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setFont(QFont("Consolas", 10))
        row1.addWidget(self._wrap("Results", self.results_box))

        self.fig_att = Figure(figsize=(5, 5))
        self.canvas_att = FigureCanvas(self.fig_att)
        self.ax_att = self.fig_att.add_subplot(111)
        row1.addWidget(self._wrap("Roll / Pitch / Yaw (deg)", self.canvas_att))

        self.fig_clcd = Figure(figsize=(5, 5))
        self.canvas_clcd = FigureCanvas(self.fig_clcd)
        self.ax_clcd = self.fig_clcd.add_subplot(111)
        row2.addWidget(self._wrap("CL vs CD", self.canvas_clcd))

        self.fig_clalpha = Figure(figsize=(5, 5))
        self.canvas_clalpha = FigureCanvas(self.fig_clalpha)
        self.ax_clalpha = self.fig_clalpha.add_subplot(111)
        row2.addWidget(self._wrap("CL vs alpha_wing", self.canvas_clalpha))

        self.fig_ld = Figure(figsize=(5, 5))
        self.canvas_ld = FigureCanvas(self.fig_ld)
        self.ax_ld = self.fig_ld.add_subplot(111)
        row3.addWidget(self._wrap("Lift & Drag (N)", self.canvas_ld))

        self.fig_m = Figure(figsize=(5, 5))
        self.canvas_m = FigureCanvas(self.fig_m)
        self.ax_m = self.fig_m.add_subplot(111)
        row3.addWidget(self._wrap("Moments Mx/My/Mz (N*m)", self.canvas_m))

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        return layout

    def _wrap(self, title, canvas):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        v.addWidget(canvas)
        return box

    def _print(self, text):
        """Prints to both the console and the GUI's Results panel."""
        print(text)
        self.results_box.append(text)

    # -------------------------------------------------------------- solve case
    def _on_solve_case(self):
        """Solves the selected named case (wind_tunnel_cases.py) end to end:
        works out the rotor speeds, tilts, and elevon deflection that actually
        satisfy that flight condition (see wind_tunnel_solver.py for the two
        solve strategies), writes ALL of them into wind_tunnel_input.py, then
        runs the clamped/free simulation with them. Nothing here is a manually
        chosen value -- every actuator setting comes from the solve."""
        self.results_box.clear()
        case_name = self.combo_case.currentText()
        case = cases.CASES[case_name]
        self._print("=" * 60)
        self._print(f"CASE: {case_name}")
        self._print(f"  {case.get('description', '')}")

        if case.get("interactive"):
            self._print("This case opens the live interactive slider tool in a separate window.")
            self._print("=" * 60)
            import wind_tunnel_live
            self._live_window = wind_tunnel_live.LiveWindTunnel()
            self._live_window.show()
            return

        result = solver.solve_case(self.params, case)
        for line in result["report"]:
            self._print(line)

        self._write_full_actuators_to_input_file(result)
        self._print(f"-> wind_tunnel_input.py updated with the solved actuator values.")
        self._print("=" * 60)

        self._on_run()

    def _write_full_actuators_to_input_file(self, result):
        import re
        path = os.path.join(scripts_dir, "wind_tunnel_input.py")
        text = open(path, "r").read()
        assignments = {
            "ROTOR_SPEED_FR": result["n_FR"], "ROTOR_SPEED_AL": result["n_AL"],
            "ROTOR_SPEED_FL": result["n_FL"], "ROTOR_SPEED_AR": result["n_AR"],
            "ROTOR_TILT_FR": result["tilt_FR"], "ROTOR_TILT_AL": result["tilt_AL"],
            "ROTOR_TILT_FL": result["tilt_FL"], "ROTOR_TILT_AR": result["tilt_AR"],
            "ELEVON_1_DEG": result["elevon1"], "ELEVON_2_DEG": result["elevon2"],
            "WIND_VELOCITY_MPS": result["wind"],
        }
        for name, value in assignments.items():
            text = re.sub(rf"^{name} = .*$", f"{name} = {value:.6f}", text, count=1, flags=re.MULTILINE)
        open(path, "w").write(text)
        importlib.reload(cfg)

    # -------------------------------------------------------------- solve trim
    def _on_solve_trim(self):
        """Given the rotor speeds and wind velocity from wind_tunnel_input.py,
        solve for the angle of attack, elevon deflection, and rotor tilt that
        balance the vehicle in steady flight at the target flight-path angle.
        alpha is constrained to be >= 0: this is a symmetric airfoil, so
        negative alpha only produces downward (useless) lift, never a valid
        trim state. Tilt is constrained to [0,90]deg, the actuator's real
        range -- no unbounded/unreachable solutions."""
        importlib.reload(cfg)
        self.results_box.clear()
        p = self.params
        V = cfg.WIND_VELOCITY_MPS
        if V <= 0.1:
            self._print("Solve Trim needs a nonzero WIND_VELOCITY_MPS in wind_tunnel_input.py.")
            return
        gamma = np.radians(cfg.TARGET_FLIGHT_PATH_ANGLE_DEG)
        mg = p.mass * p.g
        # Trim solve assumes wings-level (no roll): front pair / aft pair each averaged.
        n_front = 0.5 * (cfg.ROTOR_SPEED_FR + cfg.ROTOR_SPEED_FL)
        n_aft = 0.5 * (cfg.ROTOR_SPEED_AL + cfg.ROTOR_SPEED_AR)

        def forces_and_moment(x):
            alpha, delta_e, tilt = x
            u, w = V * np.cos(alpha), V * np.sin(alpha)
            n_arr = np.array([n_front, n_aft, n_front, n_aft])
            lam_arr = np.array([tilt, tilt, tilt, tilt])
            rotor_forces, rotor_moments, rotor_thrust = rotors.rotor_forces_moments(
                n=n_arr, lam=lam_arr, params=p, lam_dot=None, u=u, v=0.0, w=w, p=0.0, q=0.0, r=0.0)
            Tf = rotors.front_pair_thrust(rotor_thrust)
            Va, alpha_air, beta, qbar = lifting_body.airdata(u, 0.0, w, p)
            eps, qbar_wing = lifting_body.downwash(Tf, tilt, Va, qbar, p)
            alpha_wing = alpha_air - eps
            delta_e_used = elevons.apply_stall_constraint(delta_e, alpha_wing, p)
            CL, CD, Cm, Cl, Cn = lifting_body.aero_coefficients(alpha_wing, 0.0, 0.0, 0.0, beta, Va, p)
            Lclean, D, Y, M = lifting_body.clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, 0.0, Va, beta, p, alpha_wing)
            lbf, lbm = lifting_body.project_to_body(Lclean, D, Y, M, alpha_wing, beta)
            ef, em = elevons.elevon_forces_moments(delta_e_used, 0.0, alpha_wing, qbar_wing, p)
            total_force, total_moment = rotor_forces + lbf + ef, rotor_moments + lbm + em
            theta = gamma + alpha
            R = quat2rot(np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]))
            world_force = R @ total_force
            return world_force, total_moment, dict(alpha=alpha, delta_e=delta_e_used, tilt=tilt,
                Va=Va, alpha_wing=alpha_wing, eps=eps, CL=CL, CD=CD, Lclean=Lclean, D=D)

        def residuals(x):
            wf, tm, _ = forces_and_moment(x)
            return [wf[0], wf[2] + mg, tm[1]]

        # alpha floored at 0: symmetric airfoil, negative alpha is never a valid trim.
        bounds_lo = [np.radians(0.0), np.radians(-25.0), np.radians(0.0)]
        bounds_hi = [np.radians(60.0), np.radians(25.0), np.radians(90.0)]
        x0 = np.clip([np.radians(5.0), np.radians(5.0), np.radians(45.0)], bounds_lo, bounds_hi)

        sol = least_squares(residuals, x0, bounds=(bounds_lo, bounds_hi))
        wf, tm, d = forces_and_moment(sol.x)

        self._print("=" * 60)
        self._print(f"SOLVE TRIM: V={V:.2f} m/s  gamma={cfg.TARGET_FLIGHT_PATH_ANGLE_DEG:.3f} deg")
        self._print(f"  rotor speed front pair (avg)={n_front:.3f} rev/s  aft pair (avg)={n_aft:.3f} rev/s")
        if sol.cost > 1e-6:
            self._print(f"  WARNING: no clean balance found within actuator limits (cost={sol.cost:.6f}).")
            self._print(f"  This rotor speed likely can't trim at this speed/angle -- try higher RPM.")
        self._print(f"  alpha (wing AoA):     {np.degrees(d['alpha']):.4f} deg")
        self._print(f"  elevon deflection:    {np.degrees(d['delta_e']):.4f} deg")
        self._print(f"  rotor tilt:           {np.degrees(d['tilt']):.4f} deg")
        self._print(f"  alpha_wing (post-downwash): {np.degrees(d['alpha_wing']):.4f} deg "
                     f"(downwash eps={np.degrees(d['eps']):.4f} deg)")
        self._print(f"  CL={d['CL']:.5f}  CD={d['CD']:.5f}  Lift(wing)={d['Lclean']:.3f}N  Drag(wing)={d['D']:.3f}N")
        self._print(f"  residuals: horizontal={wf[0]:.6f}N  vertical={wf[2]+mg:.6f}N  moment={tm[1]:.6f}N*m")

        # Write the solved tilt/elevon straight into the input file (rotor speeds and
        # everything else stay exactly as you set them) so the very next Run uses this
        # exact force-balanced condition -- no manual copy-paste, no chance of running
        # Run against a mismatched, untrimmed tilt/elevon by accident.
        tilt_deg = np.degrees(d['tilt'])
        elevon_deg = np.degrees(d['delta_e'])
        self._write_trim_to_input_file(tilt_deg, elevon_deg)
        self._print(f"-> wind_tunnel_input.py updated: all 4 rotor tilts = {tilt_deg:.4f} deg, "
                     f"both elevons = {elevon_deg:.4f} deg")
        self._print("=" * 60)

        self._on_run()

    def _write_trim_to_input_file(self, tilt_deg, elevon_deg):
        import re
        path = os.path.join(scripts_dir, "wind_tunnel_input.py")
        text = open(path, "r").read()
        for name in ["ROTOR_TILT_FR", "ROTOR_TILT_AL", "ROTOR_TILT_FL", "ROTOR_TILT_AR"]:
            text = re.sub(rf"^{name} = .*$", f"{name} = {tilt_deg:.6f}", text, count=1, flags=re.MULTILINE)
        for name in ["ELEVON_1_DEG", "ELEVON_2_DEG"]:
            text = re.sub(rf"^{name} = .*$", f"{name} = {elevon_deg:.6f}", text, count=1, flags=re.MULTILINE)
        open(path, "w").write(text)
        importlib.reload(cfg)

    # ------------------------------------------------------------------ run
    def _on_run(self):
        importlib.reload(cfg)
        self.results_box.clear()
        self.timer.stop()
        self.btn_play.setText("Play")

        p = self.params
        plant = Interceptor(params=p, id=1)
        plant._state_vector[2] = -START_ALTITUDE_M
        plant._state_vector[3] = cfg.WIND_VELOCITY_MPS   # u: wind in -X == vehicle flying forward at this speed
        frozen_position = plant._state_vector[0:3].copy()
        world_wind_vector = plant._state_vector[3:6].copy()   # true wind direction, fixed in the WORLD frame
        clamped = cfg.CLAMPED_BALL_SOCKET

        n1 = cfg.ROTOR_SPEED_FR
        n2 = cfg.ROTOR_SPEED_AL
        n3 = cfg.ROTOR_SPEED_FL
        n4 = cfg.ROTOR_SPEED_AR
        lam1 = np.radians(cfg.ROTOR_TILT_FR)
        lam2 = np.radians(cfg.ROTOR_TILT_AL)
        lam3 = np.radians(cfg.ROTOR_TILT_FL)
        lam4 = np.radians(cfg.ROTOR_TILT_AR)
        delta1 = np.radians(cfg.ELEVON_1_DEG)
        delta2 = np.radians(cfg.ELEVON_2_DEG)
        # actuator vector layout: [elevon1, elevon2, tilt1..4, n1..4] with rotor order
        # [FR, AL, FL, AR] -- see InterceptorAllocator.py's docstring
        control_vector = np.array([delta1, delta2, lam1, lam2, lam3, lam4, n1, n2, n3, n4])
        plant._control_vector = control_vector

        dt = DT
        steps = int(cfg.RUN_DURATION_S / dt)
        log = {k: [] for k in ['t', 'x', 'y', 'z', 'u', 'v', 'w', 'roll', 'pitch', 'yaw', 'quat',
                                'Va', 'CL', 'CD', 'alpha_wing_deg', 'wing_scale',
                                'Lclean', 'D', 'Mx', 'My', 'Mz']}

        self._print("=" * 60)
        self._print(f"RUN {'[CLAMPED / ball-socket]' if clamped else '[FREE FLIGHT]'}")
        self._print(f"  rotors [FR,AL,FL,AR] speed = [{n1:.2f},{n2:.2f},{n3:.2f},{n4:.2f}] rev/s")
        self._print(f"  rotors [FR,AL,FL,AR] tilt  = [{cfg.ROTOR_TILT_FR:.2f},{cfg.ROTOR_TILT_AL:.2f},"
                     f"{cfg.ROTOR_TILT_FL:.2f},{cfg.ROTOR_TILT_AR:.2f}] deg")
        self._print(f"  elevon1/elevon2 = {cfg.ELEVON_1_DEG:.2f} / {cfg.ELEVON_2_DEG:.2f} deg   "
                     f"wind = {cfg.WIND_VELOCITY_MPS:.2f} m/s   duration = {cfg.RUN_DURATION_S:.2f} s")

        diverged_at = None
        for i in range(steps):
            t = i * dt
            plant.state_update(dt)
            if clamped:
                # Ball-socket mount: position is physically clamped (never moves), but the
                # vehicle can pivot freely -- so the wind stays fixed in the WORLD frame while
                # its representation in the (rotating) BODY frame changes with attitude. This
                # keeps the real "weathervaning" restoring effect a free-pivot mount has;
                # freezing body-frame velocity outright would wrongly hold the apparent AoA
                # constant even as the vehicle rotates through it.
                plant._state_vector[0:3] = frozen_position
                R_world_from_body = quat2rot(plant._state_vector[6:10])
                plant._state_vector[3:6] = R_world_from_body.T @ world_wind_vector
            sv = plant.get_state_vector()
            if not np.all(np.isfinite(sv)):
                diverged_at = t
                self._print(f">>> DIVERGED (NaN/Inf) at t={t:.3f}s <<<")
                break
            d = plant.last_diagnostics
            q = sv[6:10]
            row = dict(t=t, x=sv[0], y=sv[1], z=sv[2], u=sv[3], v=sv[4], w=sv[5],
                       roll=self._roll(q), pitch=self._pitch(q), yaw=self._yaw(q), quat=q.copy(),
                       Va=d['Va'], CL=d['CL'], CD=d['CD'], alpha_wing_deg=np.degrees(d['alpha_wing']),
                       wing_scale=d['wing_scale'],
                       Lclean=d['Lclean'], D=d['D'],
                       Mx=d['total_moment'][0], My=d['total_moment'][1], Mz=d['total_moment'][2])
            for k, v in row.items():
                log[k].append(v)

        if diverged_at is None:
            last = {k: log[k][-1] for k in log if k != 'quat'}
            self._print("-" * 60)
            self._print(f"Final: pos=({last['x']:.2f},{last['y']:.2f},{last['z']:.2f})")
            self._print(f"  roll/pitch/yaw = ({last['roll']:.2f}, {last['pitch']:.2f}, {last['yaw']:.2f}) deg")
            self._print(f"  CL={last['CL']:.4f}  CD={last['CD']:.5f}  alpha_wing={last['alpha_wing_deg']:.2f} deg")
            if last['wing_scale'] <= 0.0:
                self._print("  (wing INACTIVE: Va below the hover deadband -- the CL/CD above are the")
                self._print("   table's coefficients at this alpha, but the wing is producing no force)")
            elif last['wing_scale'] < 1.0:
                self._print(f"  (wing only {100*last['wing_scale']:.0f}% active -- Va still below Va_reg)")
            self._print(f"  Lift={last['Lclean']:.2f}N  Drag={last['D']:.2f}N")
            self._print(f"  Mx/My/Mz = ({last['Mx']:.4f}, {last['My']:.4f}, {last['Mz']:.4f}) N*m")
        self._print("=" * 60)

        self.log = {k: np.array(v) if k != 'quat' else v for k, v in log.items()}
        # Tilt is constant across a run (the actuator vector is held fixed), so the
        # 3D view can just carry the commanded value for its rotor discs.
        self.log['tilt'] = np.array([lam1, lam2, lam3, lam4])
        self._redraw_static_plots()

        n_frames = len(self.log['t'])
        self.slider.setEnabled(n_frames > 0)
        self.btn_play.setEnabled(n_frames > 0)
        self.slider.setMaximum(max(0, n_frames - 1))
        self.slider.setValue(0)
        self.frame = 0
        self._draw_3d_frame(0)

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

    # --------------------------------------------------------------- redraw
    @staticmethod
    def _floor_ylim(ax, min_span):
        """Keep the y-axis at least min_span wide, centred on the data.

        Without this, matplotlib autoscales a flat trace of floating-point
        zero (hover pitch drifts ~1e-4 deg over 3 s purely from the rotor
        trim constants being quoted to 5 decimals) into a full-height curve
        with a 1e-5 axis multiplier, which reads as real motion when it is
        nothing of the kind.
        """
        lo, hi = ax.get_ylim()
        if hi - lo < min_span:
            mid = 0.5 * (lo + hi)
            ax.set_ylim(mid - min_span / 2.0, mid + min_span / 2.0)

    def _redraw_static_plots(self):
        t = self.log['t']

        self.ax_att.clear()
        self.ax_att.plot(t, self.log['roll'], label='roll')
        self.ax_att.plot(t, self.log['pitch'], label='pitch')
        self.ax_att.plot(t, self.log['yaw'], label='yaw')
        self.ax_att.set_xlabel("t (s)")
        self.ax_att.legend()
        self.ax_att.grid(True)
        self._floor_ylim(self.ax_att, 2.0)      # degrees
        self.canvas_att.draw()

        self.ax_clcd.clear()
        self.ax_clcd.plot(self.log['CD'], self.log['CL'], '-o', markersize=2)
        self.ax_clcd.set_xlabel("CD")
        self.ax_clcd.set_ylabel("CL")
        self.ax_clcd.grid(True)
        self._floor_ylim(self.ax_clcd, 0.05)
        self.canvas_clcd.draw()

        self.ax_clalpha.clear()
        self.ax_clalpha.plot(self.log['alpha_wing_deg'], self.log['CL'], '-o', markersize=2)
        self.ax_clalpha.set_xlabel("alpha_wing (deg)")
        self.ax_clalpha.set_ylabel("CL")
        self.ax_clalpha.grid(True)
        self._floor_ylim(self.ax_clalpha, 0.05)
        self.canvas_clalpha.draw()

        self.ax_ld.clear()
        self.ax_ld.plot(t, self.log['Lclean'], label='Lift')
        self.ax_ld.plot(t, self.log['D'], label='Drag')
        self.ax_ld.set_xlabel("t (s)")
        self.ax_ld.set_ylabel("N")
        self.ax_ld.legend()
        self.ax_ld.grid(True)
        self._floor_ylim(self.ax_ld, 1.0)
        self.canvas_ld.draw()

        self.ax_m.clear()
        self.ax_m.plot(t, self.log['Mx'], label='Mx')
        self.ax_m.plot(t, self.log['My'], label='My')
        self.ax_m.plot(t, self.log['Mz'], label='Mz')
        self.ax_m.set_xlabel("t (s)")
        self.ax_m.set_ylabel("N*m")
        self.ax_m.legend()
        self.ax_m.grid(True)
        self._floor_ylim(self.ax_m, 0.02)
        self.canvas_m.draw()

    def _draw_3d_frame(self, idx):
        if not self.log or len(self.log['t']) == 0:
            return
        q = self.log['quat'][idx]
        tilt = self.log.get('tilt')
        airframe_view.draw_airframe(self.ax_3d, q, self.params, quat2rot,
                                     tilt=tilt, title=f"t = {self.log['t'][idx]:.3f}s")
        self.canvas_3d.draw()

    # ------------------------------------------------------------ playback
    def _on_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(20)
            self.btn_play.setText("Pause")

    def _advance_frame(self):
        if self.log is None:
            return
        n = len(self.log['t'])
        self.frame = (self.frame + 1) % n
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame)
        self.slider.blockSignals(False)
        self._draw_3d_frame(self.frame)

    def _on_slider(self, value):
        self.frame = value
        self._draw_3d_frame(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindTunnelWindow()
    window.show()
    sys.exit(app.exec())
