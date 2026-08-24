"""
3D airframe drawing for the DeltaV Interceptor, shared by wind_tunnel_gui.py
and wind_tunnel_live.py so both show the same vehicle.

Replaces the old placeholder (two crossed lines, which read as a symmetric
"+" quadcopter and looks nothing like this aircraft). Everything positional
here comes from InterceptorParams, so the picture is the geometry the physics
is actually using -- rotor stations, rotor diameter and span are not redrawn
by hand:

    rotor_x_arm      front / aft longitudinal stations about the CG
    rotor_y_arm      lateral stations -- ASYMMETRIC H-frame, aft pair wider
    rotor_diameter   disc size
    b                wing span

Orientation cues, because a flying wing at an odd attitude is genuinely hard
to read otherwise:

    NOSE      bright arrow ahead of the aircraft, labelled
    GREEN     starboard (right) wingtip   -- real navigation-light convention
    RED       port (left) wingtip
    MAST      short spar pointing out of the top of the wing, so "inverted"
              is distinguishable from "level"

Rotor discs are drawn tilted by their live lambda, so the tilt command is
visible in the picture rather than only in the numbers.

Body frame is NED: +x forward, +y right, +z DOWN. Callers invert the z axis
so up is up on screen.
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROTOR_NAMES = ("FR", "AL", "FL", "AR")   # matches params.rotor_x_arm / rotor_y_arm order

# Planform in CG-centred body coords. Proportioned so root/tip chords average
# close to the real c_bar (0.291 m) and the span matches params.b.
_NOSE_X    =  0.25
_TIP_LE_X  =  0.06
_TIP_TE_X  = -0.12
_ROOT_TE_X = -0.15

_WING_FACE  = "#8f9aa6"
_WING_EDGE  = "#39424d"
_ELEVON     = "#c0562a"
_DISC_FRONT = "#1d5d87"
_DISC_AFT   = "#4a7f5e"
_NOSE       = "#d94f2b"
_STARBOARD  = "#1faa4b"
_PORT       = "#d42f2f"
_MAST       = "#39424d"


def _tf(R, pts):
    """Body-frame points -> world frame."""
    return (R @ np.asarray(pts, dtype=float).T).T


def _te_x(y, semi):
    """Trailing-edge x at spanwise station y (mirrors for negative y)."""
    f = 1.0 - min(abs(y) / semi, 1.0)
    return _TIP_TE_X + (_ROOT_TE_X - _TIP_TE_X) * f


def _wing_polygon(semi):
    return np.array([
        [_NOSE_X,     0.0,    0.0],
        [_TIP_LE_X,   semi,   0.0],
        [_TIP_TE_X,   semi,   0.0],
        [_ROOT_TE_X,  0.0,    0.0],
        [_TIP_TE_X,  -semi,   0.0],
        [_TIP_LE_X,  -semi,   0.0],
    ])


def _elevon_polygon(semi, sign):
    """Trailing-edge control surface on one wing. sign=+1 starboard, -1 port."""
    y_out, y_in = 0.93 * semi, 0.33 * semi
    chord = 0.05
    return np.array([
        [_te_x(y_out, semi),         sign * y_out, 0.0],
        [_te_x(y_in, semi),          sign * y_in,  0.0],
        [_te_x(y_in, semi) + chord,  sign * y_in,  0.0],
        [_te_x(y_out, semi) + chord, sign * y_out, 0.0],
    ])


def _disc(center, lam, radius, n=40):
    """Rotor disc as a circle whose plane is normal to the tilted thrust axis.

    lam=0    -> disc horizontal, thrust along body -z (hover)
    lam=90   -> disc vertical, thrust along body +x (cruise)
    """
    th = np.linspace(0.0, 2.0 * np.pi, n)
    e1 = np.array([np.cos(lam), 0.0, np.sin(lam)])   # in-plane, fore/aft
    e2 = np.array([0.0, 1.0, 0.0])                   # in-plane, lateral
    return np.asarray(center, float) + radius * (np.cos(th)[:, None] * e1
                                                 + np.sin(th)[:, None] * e2)


def draw_airframe(ax, quat, params, quat2rot, tilt=None, title=None, lim=None,
                   center=None, scale=1.0, clear=True, set_limits=True):
    """Draw the vehicle at attitude `quat` onto a 3D axes.

    tilt   : scalar, length-4 array, or None -> rotor tilt angles in RADIANS.
    center : world position to place the model at (default origin).
    scale  : geometric exaggeration. A 0.6 m aircraft on a 50 m turn circle is
             invisible at true scale, so the trajectory view blows it up.
    clear / set_limits : let a caller composite the model into a bigger scene
             (a flight path, say) instead of owning the whole axes.
    quat2rot is passed in so this module stays import-light.
    """
    R = quat2rot(np.asarray(quat, dtype=float))
    semi = 0.5 * params.b
    rad = 0.5 * params.rotor_diameter
    xs = np.asarray(params.rotor_x_arm, dtype=float)
    ys = np.asarray(params.rotor_y_arm, dtype=float)
    off = np.zeros(3) if center is None else np.asarray(center, dtype=float)

    if tilt is None:
        lams = np.zeros(4)
    else:
        lams = np.full(4, float(tilt)) if np.isscalar(tilt) else np.asarray(tilt, dtype=float)

    def T(pts):
        """body coords -> scaled, rotated, translated world coords"""
        return _tf(R, np.asarray(pts, dtype=float) * scale) + off

    if lim is None:
        lim = float(max(np.max(np.abs(ys)) + rad, semi, _NOSE_X)) * 1.02

    if clear:
        ax.clear()

    # ---- wing ----------------------------------------------------------
    ax.add_collection3d(Poly3DCollection(
        [T(_wing_polygon(semi))], facecolor=_WING_FACE, edgecolor=_WING_EDGE,
        linewidths=1.2, alpha=0.85))

    # ---- elevons -------------------------------------------------------
    for sgn in (1, -1):
        ax.add_collection3d(Poly3DCollection(
            [T(_elevon_polygon(semi, sgn))], facecolor=_ELEVON,
            edgecolor=_WING_EDGE, linewidths=0.8, alpha=0.95))

    # ---- booms + rotor discs ------------------------------------------
    for i, name in enumerate(ROTOR_NAMES):
        hub = np.array([xs[i], ys[i], 0.0])
        root = np.array([xs[i] * 0.35, np.sign(ys[i]) * min(abs(ys[i]), semi * 0.55), 0.0])
        seg = T(np.vstack([root, hub]))
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=_WING_EDGE, linewidth=2.0)

        disc = T(_disc(hub, lams[i], rad))
        colour = _DISC_FRONT if xs[i] > 0 else _DISC_AFT
        ax.plot(disc[:, 0], disc[:, 1], disc[:, 2], color=colour, linewidth=1.4)
        ax.add_collection3d(Poly3DCollection([disc], facecolor=colour,
                                              edgecolor="none", alpha=0.22))

    # ---- nose arrow ----------------------------------------------------
    nose = T([[_NOSE_X, 0.0, 0.0], [_NOSE_X + 0.16, 0.0, 0.0]])
    ax.plot(nose[:, 0], nose[:, 1], nose[:, 2], color=_NOSE, linewidth=3.0)
    ax.scatter(nose[1, 0], nose[1, 1], nose[1, 2], color=_NOSE, s=30)
    if center is None:
        ax.text(nose[1, 0], nose[1, 1], nose[1, 2], "  NOSE", color=_NOSE,
                fontsize=8, fontweight="bold")

    # ---- navigation lights: green = starboard, red = port ---------------
    for sgn, colour in ((1, _STARBOARD), (-1, _PORT)):
        tip = T([[_TIP_LE_X, sgn * semi, 0.0]])[0]
        ax.scatter(tip[0], tip[1], tip[2], color=colour, s=30, depthshade=False)

    # ---- "up" mast so inverted attitudes are unambiguous ----------------
    mast = T([[0.0, 0.0, 0.0], [0.0, 0.0, -0.11]])
    ax.plot(mast[:, 0], mast[:, 1], mast[:, 2], color=_MAST, linewidth=2.0)

    # ---- axes ----------------------------------------------------------
    if set_limits:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.invert_zaxis()          # NED: +z is down, so flip for a natural view
        ax.view_init(elev=24, azim=-58)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.06)
    if title:
        ax.set_title(title, fontsize=9)
