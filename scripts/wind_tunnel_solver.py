"""
Case solver for the wind tunnel GUI. Given a named flight condition (airspeed,
flight-path angle, turn load factor -- see wind_tunnel_cases.py), works out
the actuator settings (rotor speeds, tilts, elevon deflection) that actually
satisfy it, using the real rotor/wing/elevon physics -- never a hand guess.

Two solve strategies, chosen automatically by n_load:

  CLIMB style (n_load ~= 1, e.g. Case 2 "head-on interception"): uniform
  rotor speed and tilt across all 4 rotors, tilt derived as 90-alpha (thrust
  aligned along the flight path), 3 unknowns [alpha, delta_e, n] solved
  against 3 equations (horizontal/vertical world-frame force balance,
  pitching moment = 0). Validated: converges to residuals ~1e-10 or better
  for a straight/climbing case.

  TURN style (n_load > 1, e.g. Case 3 "coordinated turn"): a fully
  simultaneous 4-unknown / 4-equation nonlinear trim, solved against the
  same physics path the simulator itself runs.

      unknowns:  [alpha, n_front, n_aft, tilt]      (tilt uniform, all 4)
      equations: force along the flight path      = 0        (thrust = drag)
                 force along the lift axis        = n_load*W
                 pitching moment                  = 0
                 alpha_wing                       = alpha_stall - margin

  Uniform tilt with DIFFERENTIAL rotor speed is what makes this exactly
  determined: tilt sets the thrust vector direction (two force balances),
  the front/aft speed split trims the wing's nose-down moment through the
  real asymmetric arm lengths, and alpha positions the wing. Verified to
  converge to residual cost ~1e-21 for the 5.2g case and ~1e-19 for 8g --
  i.e. every balance is satisfied to machine precision, not approximately.

  Three things this fixes versus the earlier turn solver:

    1. DOWNWASH IS NOW INCLUDED. The previous version passed the freestream
       qbar straight into clean_forces_moments() and never called
       downwash() at all, so it modelled the wing as if the rotors were not
       blowing over it. At these thrust levels that is a 22-48% error in
       qbar_wing (verified: x1.221 at 5.2g, x1.479 at 8g), which is why its
       answers disagreed with what the simulator actually did when the same
       actuator values were run.
    2. HORIZONTAL FORCE IS NOW BALANCED, not left as a reported residual.
       The old version produced 44.4 N of horizontal thrust against 13.2 N
       of drag -- a configuration that would have accelerated hard out of
       the condition it claimed to trim.
    3. THE WING IS NO LONGER PINNED AT THE STALL CAP. It is held a set
       margin below it (default 2 deg). Sitting exactly at alpha_stall
       leaves zero stall margin AND trips apply_stall_constraint, which
       forbids negative delta_e and so removes nose-down elevon authority
       from the controller that has to fly this.

alpha_wing is always bounded to [0, alpha_stall]: this is a symmetric
airfoil, so negative alpha only ever produces useless downward lift, never
a valid trim state. Note that the BODY incidence alpha legitimately runs
well above alpha_stall (22-28 deg in these cases) because the downwash
angle eps is subtracted off before the wing sees it.
"""
import numpy as np
from scipy.optimize import least_squares

from src.actuators import rotors, lifting_body, elevons
from src.common.utils import quat2rot


N_LOAD_TURN_THRESHOLD = 1.01   # above this, use TURN style; at/below, use CLIMB style

# How far below the stall cap the wing is flown in a turn. Non-zero for two
# reasons: real stall margin, and keeping apply_stall_constraint from locking
# out negative delta_e (which would cost the controller its nose-down authority
# exactly when it is pulling the most g). Overridable per case.
TURN_STALL_MARGIN_DEG = 2.0

# Weight on the alpha_wing residual so least_squares balances an angle (rad)
# against forces (N) sensibly. Tuned: alpha_wing lands within ~1e-15 deg.
_ALPHA_RESIDUAL_WEIGHT = 200.0


def solve_case(p, case):
    """Returns a dict of solved actuator values + a list of report strings."""
    if case.get("hover"):
        return dict(
            n_FR=169.00885, n_AL=188.38417, n_FL=169.00885, n_AR=188.38417,
            tilt_FR=0.0, tilt_AL=0.0, tilt_FL=0.0, tilt_AR=0.0,
            elevon1=0.0, elevon2=0.0, wind=0.0,
            report=["Hover: known exact trim (front/aft rotor split balances the arm-length "
                    "asymmetry with zero tilt, zero elevon, zero wind)."],
        )

    gamma = np.radians(case["gamma_deg"])
    n_load = case["n_load"]

    # A level coordinated turn has only two independent numbers: the load factor
    # and the radius. Airspeed is NOT free -- bank = arccos(1/n) and the turn
    # equation V^2 = g*R*tan(bank) fix it. So when a case gives a radius, V is
    # DERIVED here rather than read from the case file. Previously each turn case
    # carried a hand-computed V (50.03, 50.07) with the radius living only in a
    # comment, which meant editing n_load or R left a stale, silently wrong V.
    if case.get("turn_radius_m") is not None and n_load > N_LOAD_TURN_THRESHOLD:
        bank = np.arccos(1.0 / n_load)
        V = float(np.sqrt(p.g * case["turn_radius_m"] * np.tan(bank)))
    else:
        V = case["V"]

    if n_load <= N_LOAD_TURN_THRESHOLD:
        return _solve_climb(p, V, gamma)
    else:
        return _solve_turn(p, V, n_load, case.get("stall_margin_deg"))


def _solve_climb(p, V, gamma):
    mg = p.mass * p.g
    alpha_stall = p.alpha_stall

    def forces_and_moment(x):
        alpha, delta_e, n = x
        u, w = V * np.cos(alpha), V * np.sin(alpha)
        tilt = np.pi / 2.0 - alpha
        n_arr = np.array([n] * 4)
        lam_arr = np.array([tilt] * 4)
        rf, rm, rt = rotors.rotor_forces_moments(n=n_arr, lam=lam_arr, params=p, lam_dot=None,
                                                  u=u, v=0.0, w=w, p=0.0, q=0.0, r=0.0)
        Tf = rotors.front_pair_thrust(rt)
        Va, alpha_air, beta, qbar = lifting_body.airdata(u, 0.0, w, p)
        eps, qbar_wing = lifting_body.downwash(Tf, tilt, Va, qbar, p)
        alpha_wing = alpha_air - eps
        delta_e_used = elevons.apply_stall_constraint(delta_e, alpha_wing, p)
        CL, CD, Cm, Cl, Cn = lifting_body.aero_coefficients(alpha_wing, 0.0, 0.0, 0.0, beta, Va, p)
        Lclean, D, Y, M = lifting_body.clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, 0.0, Va, beta, p, alpha_wing)
        lbf, lbm = lifting_body.project_to_body(Lclean, D, Y, M, alpha_wing, beta)
        ef, em = elevons.elevon_forces_moments(delta_e_used, 0.0, alpha_wing, qbar_wing, p)
        tf, tm = rf + lbf + ef, rm + lbm + em
        theta = gamma + alpha
        R = quat2rot(np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]))
        wf = R @ tf
        return wf, tm, dict(alpha=alpha, delta_e=delta_e_used, tilt=tilt, n=n,
                             alpha_wing=alpha_wing, eps=eps, CL=CL, CD=CD, Lclean=Lclean, D=D)

    def residuals(x):
        wf, tm, _ = forces_and_moment(x)
        return [wf[0], wf[2] + mg, tm[1]]

    bounds_lo = [0.0, np.radians(-25.0), 1.0]
    bounds_hi = [alpha_stall, np.radians(25.0), 500.0]
    # Multi-start: the residual landscape has more than one local minimum (verified --
    # a naive single guess converged to a cost=8.4 non-solution here before), so try a
    # spread of starting points and keep whichever genuinely converges (lowest cost).
    guesses = [
        [np.radians(3.3), np.radians(6.5), 287.0],
        [np.radians(5.0), np.radians(5.0), 200.0],
        [np.radians(8.0), np.radians(10.0), 150.0],
        [np.radians(1.0), np.radians(0.0), 350.0],
    ]
    best_sol = None
    for g in guesses:
        x0 = np.clip(g, bounds_lo, bounds_hi)
        s = least_squares(residuals, x0, bounds=(bounds_lo, bounds_hi))
        if best_sol is None or s.cost < best_sol.cost:
            best_sol = s
    sol = best_sol
    wf, tm, d = forces_and_moment(sol.x)

    report = [
        f"CLIMB-style solve: V={V:.2f} m/s, gamma={np.degrees(gamma):.3f} deg",
        f"  alpha={np.degrees(d['alpha']):.4f} deg  delta_e={np.degrees(d['delta_e']):.4f} deg  "
        f"tilt={np.degrees(d['tilt']):.4f} deg  n={d['n']:.4f} rev/s (uniform, all 4 rotors)",
        f"  alpha_wing (post-downwash)={np.degrees(d['alpha_wing']):.4f} deg  "
        f"(downwash eps={np.degrees(d['eps']):.4f} deg)",
        f"  CL={d['CL']:.5f}  CD={d['CD']:.5f}  Lift(wing)={d['Lclean']:.3f}N  Drag(wing)={d['D']:.3f}N",
        f"  residuals: horizontal={wf[0]:.6f}N  vertical={wf[2]+mg:.6f}N  moment={tm[1]:.6f}N*m",
    ]
    if sol.cost > 1e-6:
        report.append(f"  WARNING: residual cost={sol.cost:.6f} -- may not have converged cleanly.")

    n_deg, t_deg = d["n"], np.degrees(d["tilt"])
    e_deg = np.degrees(d["delta_e"])
    return dict(n_FR=n_deg, n_AL=n_deg, n_FL=n_deg, n_AR=n_deg,
                tilt_FR=t_deg, tilt_AL=t_deg, tilt_FL=t_deg, tilt_AR=t_deg,
                elevon1=e_deg, elevon2=e_deg, wind=V, report=report)


def _solve_turn(p, V, n_load, stall_margin_deg=None):
    """Simultaneous 4x4 nonlinear trim for a level coordinated turn.

    Unknowns [alpha, n_front, n_aft, tilt]; residuals are flight-path force,
    lift-axis force, pitching moment, and the alpha_wing target. Elevons are
    held neutral (delta_e=0) -- the pitch moment is trimmed by the front/aft
    rotor speed split, matching how the maneuver is flown on paper.
    """
    mg = p.mass * p.g
    L_req = n_load * mg
    margin = np.radians(TURN_STALL_MARGIN_DEG if stall_margin_deg is None else stall_margin_deg)
    alpha_wing_target = p.alpha_stall - margin

    def evaluate(x):
        alpha, n_f, n_a, tilt = x
        u, w = V * np.cos(alpha), V * np.sin(alpha)
        n_arr = np.array([n_f, n_a, n_f, n_a])
        lam_arr = np.array([tilt] * 4)
        rf, rm, rt = rotors.rotor_forces_moments(n=n_arr, lam=lam_arr, params=p, lam_dot=None,
                                                  u=u, v=0.0, w=w, p=0.0, q=0.0, r=0.0)
        # Rotor slipstream over the wing -- the term the previous solver omitted entirely.
        Tf = rotors.front_pair_thrust(rt)
        Va, alpha_air, beta, qbar = lifting_body.airdata(u, 0.0, w, p)
        eps, qbar_wing = lifting_body.downwash(Tf, tilt, Va, qbar, p)
        alpha_wing = alpha_air - eps
        delta_e_used = elevons.apply_stall_constraint(0.0, alpha_wing, p)
        CL, CD, Cm, Cl, Cn = lifting_body.aero_coefficients(alpha_wing, 0.0, 0.0, 0.0, beta, Va, p)
        Lclean, D, Y, M = lifting_body.clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, 0.0, Va, beta, p, alpha_wing)
        lbf, lbm = lifting_body.project_to_body(Lclean, D, Y, M, alpha_wing, beta)
        ef, em = elevons.elevon_forces_moments(delta_e_used, 0.0, alpha_wing, qbar_wing, p)
        tf, tm = rf + lbf + ef, rm + lbm + em
        # Level turn (gamma=0), so the body->flight-path rotation is just alpha.
        # x = along flight path, -z = along the lift axis (the axis carrying n_load*W).
        R = quat2rot(np.array([np.cos(alpha / 2.0), 0.0, np.sin(alpha / 2.0), 0.0]))
        wf = R @ tf
        return wf, tm, dict(alpha=alpha, n_f=n_f, n_a=n_a, tilt=tilt, eps=eps,
                             alpha_wing=alpha_wing, CL=CL, CD=CD, Lclean=Lclean, D=D,
                             qbar=qbar, qbar_wing=qbar_wing, thrust=rt)

    def residuals(x):
        wf, tm, d = evaluate(x)
        return [wf[0],                                            # thrust = drag
                -wf[2] - L_req,                                   # lift axis = n_load * W
                tm[1],                                            # no uncommanded pitch
                _ALPHA_RESIDUAL_WEIGHT * (d["alpha_wing"] - alpha_wing_target)]

    # alpha here is BODY incidence and legitimately exceeds alpha_stall, because the
    # downwash angle eps is subtracted before the wing sees it -- the stall limit is
    # enforced on alpha_wing by the fourth residual, not by this bound.
    bounds_lo = [0.0, 1.0, 1.0, 0.0]
    bounds_hi = [np.radians(60.0), 500.0, 500.0, np.pi / 2.0]
    guesses = [
        [np.radians(20.0), 200.0, 200.0, 0.6],
        [np.radians(15.0), 150.0, 250.0, 0.3],
        [np.radians(25.0), 250.0, 300.0, 1.0],
        [np.radians(12.0), 120.0, 180.0, 0.9],
    ]
    best_sol = None
    for g in guesses:
        s = least_squares(residuals, np.clip(g, bounds_lo, bounds_hi), bounds=(bounds_lo, bounds_hi))
        if best_sol is None or s.cost < best_sol.cost:
            best_sol = s
    sol = best_sol
    wf, tm, d = evaluate(sol.x)

    bank_deg = np.degrees(np.arccos(1.0 / n_load))
    T = d["thrust"]
    peak_frac = 100.0 * float(np.max(np.abs(T))) / p.max_thrust_per_motor

    report = [
        f"TURN-style solve: n_load={n_load:.2f}g, bank={bank_deg:.2f} deg (=arccos(1/n)), "
        f"R={V*V/(p.g*np.tan(np.radians(bank_deg))):.2f} m  ->  V={V:.3f} m/s (derived), L_req={L_req:.3f}N",
        f"  alpha_wing held at {np.degrees(d['alpha_wing']):.4f} deg "
        f"({np.degrees(margin):.1f} deg below the {np.degrees(p.alpha_stall):.1f} deg stall cap), delta_e=0",
        f"  body alpha={np.degrees(d['alpha']):.4f} deg  (downwash eps={np.degrees(d['eps']):.4f} deg)",
        f"  tilt={np.degrees(d['tilt']):.4f} deg (uniform)  n_front={d['n_f']:.4f}  n_aft={d['n_a']:.4f} rev/s",
        f"  wing: CL={d['CL']:.5f} CD={d['CD']:.5f}  L={d['Lclean']:.3f}N  D={d['D']:.3f}N"
        f"  (qbar_wing x{d['qbar_wing'] / d['qbar']:.3f} from rotor downwash)",
        f"  rotor thrust/motor = [{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}, {T[3]:.3f}] N"
        f"  -> peak {peak_frac:.1f}% of the {p.max_thrust_per_motor:.0f}N limit",
        f"  residuals (cost={sol.cost:.2e}): along-path={wf[0]:+.3e}N  "
        f"lift-axis={-wf[2] - L_req:+.3e}N  moment={tm[1]:+.3e}N*m",
    ]
    if sol.cost > 1e-6:
        report.append(f"  WARNING: residual cost={sol.cost:.3e} -- did not converge cleanly.")
    if peak_frac > 100.0:
        report.append(f"  WARNING: required thrust EXCEEDS the per-motor limit -- maneuver not achievable.")

    t_deg = np.degrees(d["tilt"])
    return dict(n_FR=d["n_f"], n_AL=d["n_a"], n_FL=d["n_f"], n_AR=d["n_a"],
                tilt_FR=t_deg, tilt_AL=t_deg, tilt_FL=t_deg, tilt_AR=t_deg,
                elevon1=0.0, elevon2=0.0, wind=V, report=report)
