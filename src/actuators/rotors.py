"""
ROTORS -- the four tilting propellers.

PLAIN-ENGLISH OVERVIEW
----------------------
This file answers: given how fast each propeller is spinning, which way each
one is tilted, and how the aircraft is moving, what push and what twist do
the four rotors put on the airframe?

THE LAYOUT. Four rotors on an H-frame, viewed from above:

        FRONT
     1 ---- 3          1 = front-right (spins CCW)
     |      |          3 = front-left  (spins CW)
     |  CG  |
     4 ---- 2          2 = aft-left    (spins CCW)
        AFT            4 = aft-right   (spins CW)

Half spin one way and half the other on purpose: each spinning propeller
tries to twist the aircraft the opposite way (Newton's third law), so
pairing them up cancels that out. Otherwise the aircraft would just spin.

THE TILT. Each rotor can pivot between two extremes:
    lambda = 0    -> pointing straight up, like a helicopter (hover)
    lambda = 90   -> pointing forward, like an aeroplane (cruise)
Anything in between is a mix of lifting and pushing forward.

WHAT ACTUALLY HAPPENS IN HERE, in order:

  1. How fast is air already entering each propeller? (A propeller flying
     forward at speed bites less air than a stationary one.)
  2. Turn that into "advance ratio" J, then look up how efficient the
     propeller is at that J.
  3. Get thrust and torque out of it.
  4. Work out the twisting effects: thrust on a lever arm, motor reaction
     torque, and a correction for air arriving at the disc at an angle.
  5. Add gyroscopic effects (spinning things resist being tilted).

WHY "MOMENT" KEEPS APPEARING
A moment is just a twisting force -- a push times how far it acts from the
centre. A rotor on the left pushing harder than one on the right does not
just lift, it ROLLS the aircraft, and the further out it sits the more it
rolls. Mx = roll, My = pitch, Mz = yaw.

Implements Full_6DOF_RigidBody_Derivation.pdf Ch 10, cross-checked
against 6DOF.pdf Sec 5.3/6.3/7.3 and the input->state trace in
Interceptor_Input_State_Dependency_Map.pdf Sec 4.6-4.7.

Rotor layout, viewed from above (6DOF.pdf Sec 2.1):
    1: front-right, CCW  (spin sign s = +1)
    2: aft-left,    CCW  (spin sign s = +1)
    3: front-left,  CW   (spin sign s = -1)
    4: aft-right,   CW   (spin sign s = -1)

Tilt convention: lambda = 0   -> hover, thrust along -z_b (up)
                 lambda = 90deg -> cruise, thrust along +x_b (forward)
                 n_hat_i = [sin(lambda_i), 0, -cos(lambda_i)]^T

Moment-arm sign convention (r_i x F_T,i under the locked body frame
x=forward, y=right, z=down, cross-checked by three independent physical
checks -- a right rotor thrusting harder rolls right side up, a front
rotor thrusting harder pitches nose up, and a right rotor's forward-tilt
thrust yaws the nose away from that side, like a twin losing its right
engine):
    x_i = +l_front  for the front pair (ahead of CG -> positive, since
                     +x_b is forward)
    x_i = -l_aft    for the aft pair (behind CG -> negative)
    Mx = -y_i*T_i*cos(lam_i),  My = x_i*T_i*cos(lam_i),  Mz = -y_i*T_i*sin(lam_i)
"""
import numpy as np

_BLADE_LIFT_SLOPE = 2.0 * np.pi   # a, 2-D thin-airfoil lift-curve slope (oblique-flow
                                    # doc Sec 7.1) -- a theoretical constant, not a
                                    # measured/vehicle-specific parameter.


def hub_velocity(u, v, w, p_rate, q_rate, r_rate, x_i, y_i):
    """
    How fast is EACH rotor hub moving through the air?

    In plain terms: the four rotors are bolted out on arms, not at the centre
    of the aircraft. So when the aircraft rotates, they get swung around and
    end up moving faster or slower than the centre does -- exactly like
    sitting on the edge of a merry-go-round versus at the middle.

    So each hub's speed = the aircraft's speed + the extra from spinning.
    The cross terms below (r*y_i, p*y_i, q*x_i) are that "extra from
    spinning" part. In a hard turn the front and aft rotors genuinely see
    different airflow because of it.

    Full 3-vector hub velocity per rotor (oblique-flow doc Eq. 1):
    V_hub,i = V_cg + Omega x r_i, r_i = (x_i, y_i, 0).

    local_axial_inflow() below only needs the projection onto the shaft
    (n_hat has no y-component, so v/r never entered it) -- that's enough
    for the baseline CT/CQ table lookup. The oblique-flow correction
    needs the full vector: its y-component (v + r*x_i) is what lets the
    in-plane flow direction point out of the x-z plane during sideslip
    or yaw rate, which the scalar projection can't represent.
    """
    Vx = u - r_rate * y_i                      # forward speed, adjusted for yaw swinging this rotor
    Vy = v + r_rate * x_i                      # sideways speed, adjusted for yaw
    Vz = w + p_rate * y_i - q_rate * x_i       # vertical speed, adjusted for roll and pitch
    return np.stack([Vx, Vy, Vz], axis=-1)   # (4,3)


def local_axial_inflow(u, v, w, p_rate, q_rate, r_rate, lam, x_i, y_i):
    """
    Of all that hub motion, how much is going STRAIGHT INTO the propeller?

    In plain terms: a propeller only cares about the air arriving along its
    own shaft -- the component going straight through the disc. Air sliding
    sideways past it barely affects its thrust (that gets handled separately,
    further down, as the "oblique flow" correction).

    So we take the hub's full 3D velocity and keep only the part pointing
    along the shaft direction. Because the tilt only pivots about the
    aircraft's left-right axis, the shaft never has a sideways component,
    which is why v never appears in the result.

    Why it matters: a propeller flying fast into the wind is already moving
    through air that is rushing at it, so each blade meets the air at a
    shallower angle and makes less thrust. This number is what tells us that.

    V_ax,i : component of rotor i's LOCAL HUB velocity along its own thrust
    axis n_hat_i = [sin(lam_i), 0, -cos(lam_i)]. The hub isn't at the CG, so
    it sees the CG velocity (u,v,w) plus the rigid-body rotation carrying it
    around: V_hub,i = [u,v,w] + omega x r_i, r_i = (x_i, y_i, 0). Projected
    onto n_hat_i (its y-component never contributes, since n_hat has no y
    part -- tilt is single-axis about body y only):
        V_ax,i = (u - r*y_i)*sin(lam_i) - (w + p*y_i - q*x_i)*cos(lam_i)

    In an aggressive coordinated turn the front/aft and left/right rotors
    see materially different effective inflow purely from the p*y_i, q*x_i,
    r*y_i cross terms -- dropping them (as an earlier version of this
    function did, using only the CG's u,v,w) biases the thrust map used by
    the control allocator.
    """
    lam = np.asarray(lam, dtype=float)
    sin_l, cos_l = np.sin(lam), np.cos(lam)
    # Tilted forward -> the forward speed feeds the disc. Tilted up (hover)
    # -> the vertical speed does. The tilt angle decides the mix.
    return (u - r_rate * y_i) * sin_l - (w + p_rate * y_i - q_rate * x_i) * cos_l  # (4,)


def advance_ratio(V_N, n, D, params):
    """
    ADVANCE RATIO J -- the single most important number for a propeller.

    In plain terms: J compares "how fast the aircraft is flying" against "how
    fast the propeller is spinning".

        J = 0        -> stationary, propeller thrashing air on the spot.
                        Maximum thrust.
        J small      -> normal flight, thrust still good.
        J large      -> flying fast for the RPM. Each blade meets the air at
                        a shallow angle and makes little thrust.
        J very large -> the air is driving the propeller instead of the other
                        way round (windmilling) -- now it makes DRAG.

    It is like a bicycle gear: the same pedalling effort does much less for
    you once you are already going fast.

    The max(n, n_reg) is just a safety floor so that a stopped propeller does
    not make this divide by zero.

    J = V_N / (n D), Selig 2014 Eq. (25). n is floored at params.n_reg
    (rev/s) to prevent J diverging at low/zero commanded rotor speed --
    same regularization pattern as params.Va_reg for the wing rate terms.
    """
    n_safe = np.maximum(np.asarray(n, dtype=float), params.n_reg)   # never divide by a stopped propeller
    return V_N / (n_safe * D)


_WINDMILL_CT_FLOOR = -0.06   # most negative CT we allow -- how much DRAG a windmilling prop can make
_WINDMILL_EXTRAP_J = 1.0     # how far past the measured data we are willing to guess


def _extrapolate_high_J(J, table_J, table_val, floor):
    """
    What to do when the aircraft flies FASTER than the propeller data covers.

    In plain terms: our measured propeller table stops at J = 1.2, which is
    where thrust reaches zero. But the aircraft can go faster than that. The
    obvious shortcut is to just hold the last value (zero) -- but that would
    say the propellers become invisible to the airflow, producing neither
    thrust nor drag, which is nonsense. Four propeller discs held sideways in
    a 60 m/s wind definitely do something.

    What really happens past that point is the WINDMILL BRAKE state: the
    airflow starts driving the propeller round instead of the other way, and
    the propeller now acts as a brake -- negative thrust, i.e. drag.

    So instead of freezing, we continue the last slope of the curve downward
    into negative territory, but with two safety rails:
      - a floor, so a wildly out-of-range J cannot invent a huge fake force
      - a limit on how far past the data we are willing to extrapolate

    Linear continuation past the last tabulated J, floored.

    np.interp CLAMPS beyond the table, which for a propeller means CT freezes
    at exactly 0 once J passes the zero-thrust point. That is not what a real
    propeller does: past zero thrust it enters the WINDMILL BRAKE state and
    produces NEGATIVE thrust (net drag) while the airstream drives the disc.
    Clamping instead reports a rotor that is transparent to the airflow --
    e.g. at tilt=90 in a 60 m/s wind this vehicle's props sit at J = 1.75,
    well past the table's 1.2, and the old code returned exactly zero force
    from four 8-inch discs held broadside-on to a 60 m/s stream.

    The last table interval's slope is continued (CT vs J is close to linear
    through the zero-thrust crossing for a fixed-pitch prop), then floored so
    a far-out-of-range J cannot produce an unbounded fictitious force. Beyond
    _WINDMILL_EXTRAP_J past the table edge the floor simply holds -- genuine
    reverse-flow/vortex-ring aerodynamics is still not modelled, and would
    need its own data.
    """
    J = np.asarray(J, dtype=float)
    J_end, v_end = table_J[-1], table_val[-1]                        # the last point we actually measured
    slope = (table_val[-1] - table_val[-2]) / (table_J[-1] - table_J[-2])   # how steeply the curve was falling there
    over = np.clip(J - J_end, 0.0, _WINDMILL_EXTRAP_J)               # how far past the data we are (capped)
    return np.where(J > J_end, np.maximum(v_end + slope * over, floor),   # past the data: continue the slope, but not below the floor
                    np.interp(J, table_J, table_val))                     # inside the data: just read it off


def lookup_CT_CQ(J, params):
    """
    Look up how good the propeller is at this advance ratio.

    In plain terms, two numbers come back:
        CT = thrust coefficient -- how much PUSH per unit of spinning
        CQ = torque coefficient -- how much RESISTANCE the motor feels

    Both are read off measured tables against J. Below the table we hold the
    first value (correct: that is the stationary case). Above the table we
    extrapolate into the windmilling regime -- see _extrapolate_high_J for
    why simply holding the last value there is wrong.

    CT(J), CQ(J) via linear interpolation over the tables in params.py
    (Selig 2014 Sec IV.D: thrust/torque coefficients from lookup tables
    on advance ratio). Below the table's range np.interp clamps, which is
    correct (static thrust). ABOVE it the tables are continued linearly into
    the windmill-brake regime rather than clamped -- see
    _extrapolate_high_J for why clamping there is physically wrong.
    """
    CT = _extrapolate_high_J(J, params.CT_table_J, params.CT_table_val, _WINDMILL_CT_FLOOR)
    CQ = _extrapolate_high_J(J, params.CQ_table_J, params.CQ_table_val, _WINDMILL_CT_FLOOR)
    return CT, CQ


def rotor_forces_moments(n, lam, params, lam_dot=None, u=0.0, v=0.0, w=0.0, p=0.0, q=0.0, r=0.0):
    """
    THE MAIN FUNCTION: total push and twist from all four rotors.

    All four rotors are computed at once using arrays, so every variable
    below holds four numbers -- one per rotor.

    THE STEPS
      1. How much air is going into each disc?          -> V_ax
      2. Turn that into advance ratio                    -> J
      3. Look up how efficient each prop is there        -> CT, CQ
      4. Get real thrust and torque                      -> T, Q
      5. Thrust pushes AND twists (lever arms)           -> FT*, MT*
      6. Motors twist back against their props           -> MQ*
      7. Correction for air hitting the disc at an angle -> FN, MN
      8. Spinning props resist being tilted (gyroscope)  -> M_gyro
      9. Add it all up

    Parameters
    ----------
    n   : array-like, shape (4,), rotor speed in rev/s (NOT rad/s)
    lam : array-like, shape (4,), tilt angle in radians
    params : InterceptorParams
    u, v, w : vehicle CG body velocity (m/s) -- drives the per-rotor advance
              ratio J so CT/CQ vary with flight condition instead of being
              fixed constants. Default 0.0 reproduces hover-at-rest.
    p, q, r : vehicle body rates (rad/s) -- the hub is offset from the CG,
              so body rotation adds a local velocity component at each
              rotor (see local_axial_inflow). Default 0.0 (no rotation)
              reproduces the old CG-velocity-only behavior if omitted.

    Returns
    -------
    Force  : (3,) total force from all four rotors, in aircraft axes
    Moment : (3,) total twist from all four rotors [roll, pitch, yaw]
    T      : (4,) thrust of each individual rotor, in newtons

    (The commented-out dict at the bottom returns every intermediate value
    instead, which is useful when debugging a single rotor.)
    """
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    if lam_dot is None:
        lam_dot = np.zeros_like(lam)   # tilt not moving right now
    lam_dot = np.asarray(lam_dot, dtype=float)

    # Where each rotor sits relative to the centre of gravity, and which way
    # it spins. These are the lever arms that turn thrust into twist.
    x_i = params.rotor_x_arm                 # (4,), how far FORWARD (+) or AFT (-) of the CG
    y_i = params.rotor_y_arm                 # (4,), how far RIGHT (+) or LEFT (-) of the CG
    s_i = params.rotor_spin_sign             # (4,), +1 = anticlockwise, -1 = clockwise

    # ---- Steps 1-4: from spin speed to actual thrust ---------------------
    V_ax = local_axial_inflow(u, v, w, p, q, r, lam, x_i, y_i)   # air entering each disc
    J = advance_ratio(V_ax, n, D=params.rotor_diameter, params=params)   # the "gear ratio"
    CT, CQ = lookup_CT_CQ(J, params)                             # how efficient each prop is right now

    # The classic propeller formulas. Thrust goes as the SQUARE of the spin
    # speed -- double the RPM, four times the thrust -- and as the fourth
    # power of diameter, which is why a slightly bigger prop helps so much.
    T = CT * params.rho * n**2 * params.rotor_diameter**4                # thrust, newtons
    Q = CQ * params.rho * n**2 * params.rotor_diameter**5                # torque the motor must supply, newton-metres

    sin_l = np.sin(lam)
    cos_l = np.cos(lam)

    # ---- Step 5a: which way does the thrust point? -----------------------
    # Tilted flat forward (lam=90): all push goes forward.
    # Pointing up (lam=0): all push goes up (negative z, since z points down).
    FTx = T * sin_l                          # forward component of thrust
    FTz = -T * cos_l                         # upward component (negative because z is down)

    # ---- Step 5b: thrust on a lever arm = twist --------------------------
    # A front rotor pushing harder pitches the nose up. A right rotor pushing
    # harder rolls the aircraft. A right rotor pushing FORWARD yaws the nose
    # left -- like a twin-engined aeroplane losing its right engine.
    MTy = x_i * T * cos_l                    # PITCH: front/aft thrust difference
    MTx = -y_i * T * cos_l                   # ROLL: left/right thrust difference
    MTz = -y_i * T * sin_l                   # YAW: from the forward-tilted component

    # ---- Step 6: the motor's reaction torque -----------------------------
    # Newton's third law: spinning a propeller one way twists the aircraft the
    # other way. s_i is what makes the anticlockwise and clockwise props
    # cancel each other instead of adding up.
    MQx = -s_i * Q * sin_l                   # reaction torque, roll component
    MQz = s_i * Q * cos_l                    # reaction torque, yaw component

    # ---- Step 7: air arriving at the disc SIDEWAYS ------------------------
    #
    # PLAIN VERSION: everything above assumed air goes straight through the
    # propeller. During transition (tilting from hover to cruise) it does not
    # -- it arrives partly across the disc. That makes the blade coming
    # forward into the wind bite harder than the one going back, so the disc
    # gets pushed sideways and twisted. On aeroplanes this is called P-factor.
    #
    # It vanishes completely whenever the flow IS straight through, which is
    # true in steady hover and in trimmed cruise.
    #
    # ---- oblique-flow correction (Sec 7): normal force + P-factor moment,
    # from the in-plane component of the hub velocity the axial table can't
    # see. Vanishes identically whenever the disc sees pure axial flow.
    n_hat = np.stack([sin_l, np.zeros_like(lam), -cos_l], axis=-1)        # each shaft's pointing direction
    V_hub = hub_velocity(u, v, w, p, q, r, x_i, y_i)                      # each hub's full 3D velocity
    # Take the full velocity, subtract the part going straight through the
    # disc, and what's left is the part sliding ACROSS it.
    V_perp_vec = V_hub - V_ax[:, None] * n_hat                           # the sideways-across-the-disc part
    V_perp_mag = np.linalg.norm(V_perp_vec, axis=-1)                     # how strong that sideways flow is
    V_perp_safe = np.maximum(V_perp_mag, 1e-9)
    d_hat = V_perp_vec / V_perp_safe[:, None]                            # which direction it points
                                                                           # (undefined at V_perp=0 but alpha_i=0
                                                                           #  there too, so it's multiplied out)

    # How obliquely is the air hitting the disc? 0 = straight through.
    #
    # Eq. 6, disc incidence. Forced to exactly 0 (not atan2's own value)
    # when V_perp is negligible: atan2(0, V_ax) returns pi, not 0, if V_ax
    # happens to be negative (reverse axial flow) -- but Sec 5 defines the
    # trigger for a nonzero correction as V_perp,i != 0, not V_ax's sign,
    # so pure-axial reverse flow must give alpha_i=0 like pure-axial
    # forward flow does, not a spurious 180 deg incidence.
    alpha_i = np.where(V_perp_mag < 1e-9, 0.0, np.arctan2(V_perp_mag, V_ax))

    # SAFETY RAMP 1. An ANGLE can be large even when the actual sideways flow
    # is microscopic -- atan2(0.000001, 0) is a full 90 degrees. Without this
    # ramp, a rounding error at hover could point this correction in a random
    # direction at near-full strength. So we also scale by how BIG the
    # sideways flow is, not just its angle.
    #
    # Magnitude-based ramp, separate from alpha_i itself: alpha_i=atan2(V_perp,V_ax)
    # is an ANGLE, so it can sit at a full 90 deg even when V_perp is only a
    # tiny fraction of a m/s (whenever V_ax happens to be ~0 too, e.g. at
    # hover) -- atan2(tiny_positive, 0) is exactly 90 deg regardless of how
    # tiny "tiny_positive" is. That let a sub-mm/s numerical perturbation in
    # V_perp flip d_hat's direction and hand the oblique correction a
    # near-full-strength force pointed in a numerically-arbitrary direction
    # -- verified by direct finite-difference linearization at a hover trim
    # point: d(u_dot)/du came out to ~1600/s, an aerodynamically impossible
    # value, traced to exactly this. Scaling PN/NP by V_perp_mag's own
    # magnitude (not its angle) below the same Va_reg floor used everywhere
    # else forces the whole correction to zero out smoothly as V_perp -> 0,
    # regardless of what alpha_i's angle happens to be doing there.
    perp_ramp = np.clip(V_perp_mag / params.Va_reg, 0.0, 1.0)
    Omega_i = 2.0 * np.pi * np.maximum(n, params.n_reg)                   # spin rate in radians/sec
    R = params.rotor_diameter / 2.0                                       # rotor radius
    # mu compares the sideways flow against the blade tip speed. Small = the
    # blades are spinning far faster than the aircraft is moving = model valid.
    mu_i = V_perp_mag / (Omega_i * R)                                    # Eq. 5, validity check

    # SAFETY RAMP 2. This correction is a small-angle approximation, only
    # trustworthy while mu stays below about 0.3. Past that it is being used
    # outside what it was derived for, so it is faded out to zero by mu = 0.4
    # rather than allowed to grow without limit.
    #
    # mu_i validity fade: this module's own docstring says the oblique-flow
    # correction (PN/NP below) is "reliable roughly below 0.3-0.4" -- but
    # nothing previously enforced that. q_i (the disc dynamic pressure PN/NP
    # scale with) grows as V_hub^2 with NO ceiling, so once a rotor is pushed
    # into a regime the linear model was never validated for (e.g. the
    # vehicle moving much faster than the rotor's own tip speed), PN/NP grow
    # without bound even while the primary CT(J)-table thrust has correctly
    # dropped to ~0 -- producing a real, unbounded, unphysical accelerating
    # force from a term that's supposed to be a small in-plane correction.
    # Confirmed: at a validated 50 m/s trim point mu_i~=0.03 (deep in the
    # trusted region, ramp has zero effect); pushed into an out-of-envelope
    # test (extreme tilt, high wind, open-loop, no controller) mu_i crosses
    # 0.3 right around where speed was still physically reasonable and is
    # far past 1.0 soon after -- exactly where this correction should stop
    # being trusted, matching how CT/CQ already fade at the table's own edge.
    mu_validity = np.clip(1.0 - (mu_i - 0.3) / (0.4 - 0.3), 0.0, 1.0)

    V_hub_mag = np.linalg.norm(V_hub, axis=-1)
    V_hub_safe = np.maximum(V_hub_mag, params.Va_reg)                     # prevents q_i=0 at rest (hover-at-rest
                                                                           # start state), same floor pattern as
                                                                           # lifting_body.py's 1/Va terms
    q_i = 0.5 * params.rho * V_hub_safe**2                                # how hard air hits this disc
    A = params.rotor_disk_area
    sigma = params.blade_solidity                                         # what fraction of the disc is actually blade
    Cd = params.blade_Cd                                                  # blade drag coefficient

    # These formulas have 1/J in them, which explodes as J approaches zero
    # (hover). So J is floored ONLY inside those specific terms -- the real,
    # unfloored J is still used everywhere it is safe.
    #
    # J floored ONLY inside the 1/J, pi/J singular terms below (Sec 7.5:
    # this linear model "cannot be reliably applied near hover" and
    # explicitly blows up as J->0) -- the real, unfloored J is still used
    # everywhere else (Cl_bar's leading J/pi*T term, which itself -> 0 as
    # J->0 and needs no floor).
    J_sign = np.where(J >= 0.0, 1.0, -1.0)
    J_safe = J_sign * np.maximum(np.abs(J), params.J_reg)

    # Average lift coefficient of the blades themselves.
    Cl_bar = (3.0 * J / (2.0 * np.pi)) * (
        (2.0 / (sigma * q_i * A)) * (J / np.pi) * T + Cd)                 # Eq. 11 (corrected form, Sec 7.2)

    # PN = sideways FORCE on the disc from the oblique flow.
    PN = (sigma * q_i * A / 2.0) * (
        Cl_bar
        + (_BLADE_LIFT_SLOPE * J / (2.0 * np.pi)) * np.log(1.0 + (np.pi / J_safe) ** 2)
        + (np.pi / J_safe) * Cd
    ) * alpha_i * perp_ramp * mu_validity                                  # both safety ramps applied here

    # NP = the TWIST that goes with it (classic P-factor).
    NP = -(sigma * q_i * A * R / 2.0) * (
        (2.0 * np.pi / (3.0 * J_safe)) * Cl_bar
        + (_BLADE_LIFT_SLOPE / 2.0) * (1.0 - (J_safe / np.pi) ** 2 * np.log(1.0 + (np.pi / J_safe) ** 2))
        - (np.pi / J_safe) * Cd
    ) * alpha_i * perp_ramp * mu_validity                                  # and here

    FN = PN[:, None] * d_hat                                              # point that force along the sideways-flow direction
    MN = s_i[:, None] * NP[:, None] * d_hat                               # spin direction decides which way the twist goes

    Force = np.zeros(3)
    Moment = np.zeros(3)

    # ---- Step 8: gyroscopic effect ---------------------------------------
    # A spinning propeller behaves like a gyroscope: try to tilt it and it
    # pushes back at 90 degrees to the way you pushed. The shaft is being
    # rotated both by the aircraft manoeuvring AND by the tilt mechanism
    # itself, so both are included.
    omega_body = np.array([p, q, r])                       # how fast the aircraft is rotating
    tilt_axis = np.asarray(params.rotor_tilt_axis, dtype=float)
    # omega_shaft = omega_body + (lam_dot * tilt_axis) for all 4 rotors
    omega_shaft = omega_body[None, :] + lam_dot[:, None] * tilt_axis[None, :]   # total rotation each shaft feels
    M_gyro = gyroscopic_moment(omega=omega_shaft, n=n, lam=lam, params=params)

    # ---- Step 9: add up all four rotors ----------------------------------
    # np.sum collapses the four per-rotor values into one total for the aircraft.
    Force[0] = np.sum(FTx + FN[:, 0])        # total forward force
    Force[1] = np.sum(FN[:, 1])              # total sideways force (only from the oblique correction)
    Force[2] = np.sum(FTz + FN[:, 2])        # total vertical force

    # Each axis collects: thrust lever arm + oblique correction + motor
    # reaction torque + gyroscopic. Pitch has no reaction-torque term because
    # the shafts tilt about the pitch axis, so their torque has no pitch part.
    Moment[0] = np.sum(MTx + MN[:, 0] + MQx) + M_gyro[0]    # total ROLL
    Moment[1] = np.sum(MTy + MN[:, 1])       + M_gyro[1]    # total PITCH
    Moment[2] = np.sum(MTz + MN[:, 2] + MQz) + M_gyro[2]    # total YAW

    return Force, Moment, T
    # return {"T": T, "Q": Q, "FTx": FTx, "FTz": FTz,
    #         "MTx": MTx, "MTy": MTy, "MTz": MTz, "MQx": MQx, "MQz": MQz,
    #         "FNx": FN[:, 0], "FNy": FN[:, 1], "FNz": FN[:, 2],
    #         "MNx": MN[:, 0], "MNy": MN[:, 1], "MNz": MN[:, 2],
    #         "J": J, "CT": CT, "CQ": CQ, "alpha_i": alpha_i, "mu_i": mu_i}


def gyroscopic_moment(omega, n, lam, params):
    """
    The gyroscope effect from four spinning propellers.

    In plain terms: hold a spinning bicycle wheel by its axle and try to tilt
    it -- it fights you, and it pushes in a direction 90 degrees away from
    where you pushed. Four spinning propellers do exactly the same to the
    aircraft whenever it manoeuvres.

    Each rotor stores angular momentum (h) proportional to how fast it spins.
    Rotating that stored momentum produces a twist, which is the cross
    product below.

    WHY THE SPIN SIGN MATTERS: two rotors on the same tilt axis point their
    shafts the same way, but if one spins clockwise and the other
    anticlockwise their stored momentum points in OPPOSITE directions. Leave
    s_i out and they would wrongly reinforce each other instead of largely
    cancelling.

    M_gyro = -omega x h_rot,   h_rot = sum_i Ip * (2*pi*n_i) * s_i * n_hat_i

    s_i (spin sign) matters here: n_hat_i is the SHAFT direction, common to
    both rotors of a tilt pair, but a CCW and a CW rotor spinning about the
    same shaft direction carry angular momentum in OPPOSITE senses. Without
    s_i, a CCW/CW pair would wrongly add instead of partially cancelling.

    omega : array-like (3,) = [p, q, r]
    """
    # omega is now expected to be a (4, 3) array
    omega = np.asarray(omega, dtype=float)
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    n_hat = np.stack([np.sin(lam), np.zeros_like(lam), -np.cos(lam)], axis=-1)  # shaft direction of each rotor
    Omega_i = 2.0 * np.pi * n                            # rev/s -> radians/s
    s_i = params.rotor_spin_sign

    # Angular momentum of each rotor: inertia x spin rate, pointing along its
    # shaft, with the sign set by which way it turns.
    h_rot_i = params.Ip * (Omega_i * s_i)[:, None] * n_hat

    # Rotating a stored angular momentum produces a twist at right angles to both.
    M_gyro_i = -np.cross(omega, h_rot_i)

    # Sum the 4 independent moments to get the net effect on the rigid body
    return np.sum(M_gyro_i, axis=0)


def front_pair_thrust(T):
    """
    Add up just the two FRONT rotors' thrust.

    In plain terms: this is the handover point between this file and the wing
    file. Only the front rotors blow air over the wing, so lifting_body.py
    needs to know how hard those two are working in order to compute the
    downwash. The aft rotors blow past the back and do not matter for that.

    Careful with the indexing: rotors are numbered 1-4 but Python arrays
    start at 0, so front-right is T[0] and front-left is T[2].

    Tf = T1 + T3 (front-right + front-left). 6DOF.pdf Sec 5.3 / Doc4 Sec 12.2.

    NOTE: array index 0 = rotor 1, index 2 = rotor 3 (see module docstring
    for the rotor-index layout).
    """
    T = np.asarray(T)
    return T[0] + T[2]   # index 0 = front-right, index 2 = front-left
