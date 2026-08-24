"""
LIFTING BODY -- the wing itself.

PLAIN-ENGLISH OVERVIEW
----------------------
This file answers one question: given how fast the aircraft is moving and
which way it is pointing, how much lift, drag and twisting force does the
WING produce? (The rotors are in rotors.py, the flaps in elevons.py.)

It happens in five steps:

  1. AIR DATA          How fast is the air going past us, and at what angle
                       is it hitting the wing?
                          -> airdata()

  2. ROTOR WASH        The propellers blow air over the wing. That air is
                       both faster AND pushed downward, so the wing sees a
                       different speed and a different angle than the
                       freestream would suggest.
                          -> downwash()

  3. COEFFICIENTS      At that angle, how "lifty" and how "draggy" is the
                       wing? Comes from real XFLR5 data where we have it,
                       and a formula where we don't. Stall is applied here.
                          -> aero_coefficients() / _lift_drag_coefficients()

  4. REAL FORCES       Turn those dimensionless numbers into actual newtons
                       and newton-metres.
                          -> clean_forces_moments()

  5. ROTATE            Convert from "relative to the airflow" into "relative
                       to the aircraft", which is what the physics engine
                       needs.
                          -> project_to_body()

TWO IDEAS THAT COME UP REPEATEDLY
---------------------------------
* DYNAMIC PRESSURE (qbar = 0.5 * rho * V^2). Think of it as "how hard the
  air is hitting you". Every force in this file is basically
  qbar * area * some_coefficient. Double the speed, quadruple the force.

* ANGLE OF ATTACK (alpha). The angle between where the wing is pointing and
  where the air is actually coming from. Not the same as the aircraft's
  pitch angle! Tilt the wing more into the wind and you get more lift --
  until it stalls.

A NOTE ON THE LONG COMMENTS
---------------------------
Several functions below have long explanations about "regularisation" or
numerical floors. Those are not academic. Each one documents a real bug that
made the simulator produce nonsense, with the measured symptom recorded so
nobody removes the fix later thinking it is unnecessary padding.
"""
import numpy as np

# Below this airspeed we call it "not really flying". A wing with no air
# moving over it does nothing, so its whole contribution gets switched off.
_WING_VA_DEADBAND = 0.5  # m/s; below this, treat as true hover -- see clean_forces_moments()

# Where the stall "kicks in" relative to the quoted stall angle.
#
# PLAIN VERSION: alpha_stall is defined as the angle where lift is at its
# MAXIMUM. If we centred the stall fade exactly there, we would already have
# thrown away half the lift at the very point where lift is supposed to peak.
# Shifting the fade 2 degrees later makes the peak of the final curve land
# exactly on alpha_stall, which is what a real wing does.
#
# Stall is applied a little ABOVE alpha_stall rather than centred on it. alpha_stall
# is by definition where CL PEAKS, so a blend centred there would put sigma=0.5 at
# the peak and cut CL_max roughly in half (verified: 0.579 -> 0.339 at 13 deg). This
# offset places the blend so the peak of the resulting curve lands ON alpha_stall,
# which is what a real lift curve does.
_POST_STALL_OFFSET = np.radians(2.0)


def _flat_plate_CN(params):
    """
    How much force a FULLY STALLED wing makes, treated as a flat plate.

    In plain terms: once a wing is stalled badly enough it stops behaving
    like a wing and starts behaving like a barn door held up to the wind.
    This number is "how draggy a barn door of this shape is".

    A very long, thin plate (infinite wingspan) scores about 2.0. A short,
    stubby one scores much less, because air escapes around the tips instead
    of piling up against the surface. This wing is stubby (aspect ratio 2.3),
    so it comes out around 1.15.

    Post-stall normal-force coefficient, corrected for finite aspect ratio.

    A flat plate broadside to the flow only reaches CD ~= 2 in the 2-D /
    infinite-span limit. A finite wing sheds flow around the tips, which
    relieves the pressure difference and lowers it a lot: Hoerner's data give
    CD_90 ~= 1.17-1.20 around AR = 1-2, rising slowly with span. The standard
    correlation CD_max = 1.11 + 0.018*AR is used here, giving ~1.15 for this
    vehicle's AR = 2.31.

    Using the 2-D value 2.0 for this wing would be wrong twice over: it
    overstates deep-stall drag by ~75%, and because the post-stall lift is
    CN*sin(a)*cos(a) it puts the post-stall CL peak at 1.0 -- ABOVE the
    attached CL_max of ~0.62 -- which would mean the wing makes more lift
    stalled than unstalled.
    """
    return 1.11 + 0.018 * params.AR


def airdata(u, v, w, params):
    """
    STEP 1: work out the basic facts about the air hitting the aircraft.

    Inputs are the aircraft's own velocity, split into its three directions:
        u = forward (out the nose)
        v = sideways (out the right wing)
        w = downward (through the floor)

    Outputs:
        Va    total airspeed -- how fast the air is going past us overall
        alpha angle of attack -- is the air hitting us from below (nose high)
              or from above (nose low)? Comes from comparing w against u.
        beta  sideslip angle -- is the air hitting us from the side rather
              than straight down the nose? (Like a car skidding sideways.)
        qbar  dynamic pressure -- how hard the air is hitting us.

    WHY THE REST OF THIS FUNCTION LOOKS COMPLICATED
    -----------------------------------------------
    The natural way to get alpha is atan2(w, u) -- "the angle between the
    forward and downward speeds". That works fine in normal flight but breaks
    badly when the aircraft is nearly stationary (hovering), because when both
    u and w are almost zero, the ANGLE between them is meaningless. A puff of
    numerical noise can swing it by 100+ degrees, and since alpha drives lift,
    that produces a huge fake force out of nothing.

    Two guards fix it, both documented in detail below:
      - u_floor: stops the maths dividing by something that is exactly zero
      - the blend: fades smoothly to a safe fixed reference at low speed

    alpha is regularized below params.Va_reg -- atan2(w,u) is a genuine
    mathematical singularity at u=w=0: the DIRECTION of a near-zero
    vector is not physically meaningful, so a fraction of a m/s of noise
    in u/w there can swing alpha by 100+ degrees (verified: a 0.1 m/s
    perturbation at hover moved alpha_wing from -85deg to +95deg). Fed
    into CL/Cm this produces a real, large, sign-flipping force/moment
    for an imperceptible velocity change -- confirmed to be exactly what
    was driving hover open-loop runs unstable (q diverging within a
    single 0.01s RK4 step even starting from an exact static trim,
    despite the trim's own residual being ~1e-9).

    Below Va_reg, alpha is a LINEAR BLEND (not a hard switch) between the
    ordinary atan2(w,u) and a regularized atan2(w,Va_reg) reference --
    same floor concept Va_reg already applies to every 1/Va rate-damping
    term elsewhere in this file, just applied to alpha's own denominator
    here. An early version used a hard if/else switch at Va=Va_reg
    instead of this blend; that has its own discontinuity right at the
    switch-over whenever w/u's ratio there differs from w/Va_reg's. The
    blend is continuous everywhere by construction -- at Va=0 it's pure
    atan2(w,Va_reg) (bounded, smooth in w), converging smoothly to the
    ordinary atan2(w,u) as Va reaches Va_reg.

    u is additionally floored towards (not clamped to) 0 before the raw
    atan2 term: atan2(w,u) has a genuine branch cut at w=0, u<0
    (atan2(0,+tiny)=0 but atan2(0,-tiny)=+-pi -- a real, unavoidable jump,
    not a smoothness artifact of this file's own regularization).
    Verified by direct linearization at a hover trim point: with u used
    raw, d(u_dot)/du came out to ~1622/s -- a physically absurd
    aerodynamic derivative -- traced to exactly this jump multiplying a
    tiny finite-difference step.

    A HARD clamp (max(u, 0.0)) removes that jump but plants an equally
    real one in its place: whenever u sits at/below 0 for any stretch of
    flight (unremarkable for a VTOL moving mostly vertically, not
    forward -- e.g. climbing, or recovering from a pitch excursion),
    atan2(w, 0) is not a smooth near-90-degree value, it is EXACTLY +/-90
    deg decided purely by sign(w) -- so alpha snaps a full 180 degrees
    the instant w crosses zero, confirmed directly: w going from
    +0.0008 to -0.0058 m/s (a functionally zero change) flipped alpha
    from exactly +90.000 to exactly -90.000 deg. u_floor below replaces
    the hard clamp with a smooth one (a softplus-style smoothed max(u,0))
    so atan2's second argument is never exactly 0 -- for any meaningfully
    forward u it's indistinguishable from plain u (no change to normal
    forward-flight alpha), and near/below u=0 it settles smoothly to a
    small positive floor instead of exactly zero, so alpha varies
    continuously through a w sign change instead of jumping.
    """
    rho, Va_reg = params.rho, params.Va_reg
    Va = float(np.sqrt(u**2 + v**2 + w**2))                   # total speed through the air (Pythagoras on the 3 components)
    if Va > 1e-6:
        beta = float(np.arcsin(np.clip(v / Va, -1.0, 1.0)))   # sideslip: how much of our speed is sideways
    else:
        beta = 0.0                                            # standing still -> no meaningful sideslip direction
    blend = min(Va / Va_reg, 1.0)                             # 0 when stationary -> 1 once properly flying
    alpha_reg = float(np.arctan2(w, Va_reg))                  # "safe" alpha: uses a fixed denominator, so it can never blow up
    u_smooth_eps = 1.0                                        # m/s; how wide the smoothing around u=0 is
    u_floor = 0.5 * (u + np.sqrt(u**2 + u_smooth_eps**2))     # a SMOOTH version of max(u, 0) -- never lands exactly on 0
    alpha_raw = float(np.arctan2(w, u_floor)) if Va > 1e-9 else 0.0   # the normal, correct alpha for real flight
    alpha = (1.0 - blend) * alpha_reg + blend * alpha_raw     # slide between "safe" (slow) and "real" (fast)
    qbar = 0.5 * rho * Va**2                                  # dynamic pressure: how hard the air is hitting us
    return Va, alpha, beta, qbar


def wing_activity(Va, params):
    """
    Is the wing actually doing anything right now? Returns 0 to 1.

    In plain terms: a wing sitting still in a hangar produces no lift. As
    airspeed builds, it gradually starts working. This returns 0 when we are
    effectively stationary, 1 once we are properly flying, and a fraction in
    between.

    It is a single shared definition on purpose, so that what the simulator
    DISPLAYS and what it actually APPLIES can never disagree about whether
    the wing is flying.

    How much of the wing's aerodynamics is live at this airspeed: 0 below the
    hover deadband, ramping to 1 by Va_reg. Single definition, used both by
    clean_forces_moments() and by the diagnostics, so reported coefficients and
    applied forces can never disagree about whether the wing is flying.
    """
    return 0.0 if Va < _WING_VA_DEADBAND else min(Va / params.Va_reg, 1.0)


def downwash(Tf, lam_f, Va, qbar, params):
    """
    STEP 2: account for the propellers blowing air over the wing.

    In plain terms: the front rotors act like fans pointed at the wing. This
    does two things at once:

      1. The air over the wing is FASTER than the freestream, so the wing
         works harder than its airspeed alone would suggest
                 -> qbar_wing is bigger than qbar

      2. That air has been pushed DOWNWARD, so it meets the wing at a
         shallower angle than the freestream does
                 -> the wing's effective angle is reduced by epsilon

    This is why the aircraft can sit at 28 degrees nose-up while the wing
    itself only sees 11 degrees: the other 17 degrees is downwash.

    THE TILT GATE (the cos(lam_f) term)
    -----------------------------------
    Picture the front rotors as a fan:
      - Pointed straight up (hover): everything it moves goes straight down
        through the wing -> full downwash.
      - Tilted flat forward (cruise): it blows air backwards past the
        fuselage, missing the wing -> almost no downwash.
    cos(lam_f) slides smoothly between those two.

        Tf_perp   = Tf * cos(lam_f)                            -- only the part of front thrust still pointing through the wing
        wi        = sqrt(Tf_perp / (2 rho Adisk))               -- induced velocity from that vertical component
        epsilon   = k_eps * wi / Va                             -- effective alpha shift from the downwash
        qbar_wing = qbar * (1 + k_q * Tf_perp / (qbar * Adisk)) -- dynamic pressure seen by the wing after downwash

    TILT GATE: picture the front rotors as a fan. Pointed straight up
    (hover, lam_f=0), all the air they move gets blown straight down onto
    the wing below -- full downwash (cos(0)=1, so Tf_perp=Tf, identical to
    the ungated formula). Tilted almost flat forward (cruise, lam_f~90deg),
    the fan blows air backward past the fuselage, not down through the
    wing -- cos(lam_f) fades Tf_perp to ~0, and this reduces to qbar_wing
    ~= qbar, eps ~= 0. Without this gate, hover-strength downwash gets
    applied even at cruise tilt, which can flip the sign of alpha_wing
    entirely (verified: alpha_wing came out at -4.93 deg instead of the
    expected +3.3 deg for a 50 m/s cruise case before this gate was added).

    NOTE: at hover, Va is near zero while wi (from the rotor thrust) is
    not, so epsilon = k_eps*wi/Va would blow up without a limit. We cap
    it at +/-85 deg to prevent that -- but below _WING_VA_DEADBAND (true
    hover, no real airspeed) we go further and force eps to exactly 0
    instead: clean_forces_moments()'s wing_scale already zeroes out the
    wing's entire force/moment contribution in this same regime (a real
    wing does nothing with no real airflow over it), so the 85 deg cap's
    own justification -- "fine because qbar_wing dominates instead of
    alpha" -- no longer holds once qbar_wing's effect is itself zeroed.
    Reporting alpha_wing as an 85-degrees-off value that does nothing is
    just confusing; alpha (~0 at true hover, no real velocity direction)
    is the honest answer once the wing has no effect either way.
    qbar_wing has no blow-up problem and needs no cap of its own.
    """
    Adisk = params.rotor_disk_area
    # Only the part of the thrust still aimed DOWN through the wing counts.
    # max(..., 0) because thrust pulling the other way cannot "un-blow" the wing.
    Tf_perp = max(float(Tf) * np.cos(lam_f), 0.0)                # the downward share of the front rotors' thrust
    # Momentum theory: push this much thrust through this much disc area and
    # the air ends up moving this fast.
    wi = np.sqrt(Tf_perp / (2.0 * params.rho * Adisk))           # how fast the rotors are pushing air downward
    eps_cap = np.radians(85.0)                                   # hard ceiling so the division below can never explode
    if Va < _WING_VA_DEADBAND:
        eps = 0.0                                                # sitting still: no meaningful flow angle to bend
    else:
        # Ratio of "downward push" to "forward speed" = how far the flow gets bent.
        eps = np.clip(params.k_eps * wi / max(Va, params.Va_reg), -eps_cap, eps_cap)  # the downwash angle
    if qbar > 1e-9:
        qbar_wing = qbar * (1.0 + params.k_q * Tf_perp / (qbar * Adisk))  # flying: freestream pressure PLUS the rotors' extra push
    else:
        qbar_wing = params.k_q * Tf_perp / Adisk                          # hovering: the ONLY air over the wing is what the rotors make
    return eps, qbar_wing


def stall_blend_sigma(alpha, alpha_s, M):
    """
    A smooth on/off switch for "has the wing stalled?" Returns 0 to 1.

    In plain terms: real wings do not snap from flying to stalled at one
    exact angle -- it comes on progressively. This returns:
        0    = flying normally, air attached
        0.5  = right in the middle of the stall break
        1    = fully stalled

    It handles BOTH directions (nose too far up AND nose too far down), which
    is what the two separate terms z1 and z2 are for. M controls how abruptly
    the change happens: bigger M = sharper stall.

    The clipping at +/-50 only stops the exponentials overflowing at extreme
    angles; it has no effect near the actual stall.

    Smoothly blends between pre-stall and post-stall lift behavior:
    close to 0 well before the stall angle, close to 1 well past it.

    The exponent is clipped just to avoid overflow at extreme alpha --
    doesn't affect the blend near the actual stall region.
    """
    z1 = np.clip(-M * (alpha - alpha_s), -50.0, 50.0)   # stall going nose-UP past +alpha_s
    z2 = np.clip(M * (alpha + alpha_s), -50.0, 50.0)    # stall going nose-DOWN past -alpha_s
    num = 1.0 + np.exp(z1) + np.exp(z2)
    den = (1.0 + np.exp(z1)) * (1.0 + np.exp(z2))
    return num / den   # 0 = attached/flying, 1 = fully stalled


def aero_coefficients(alpha_wing, p_rate, q_rate, r_rate, beta, Va, params):
    """
    STEP 3: the wing's dimensionless "how much" numbers at this angle.

    In plain terms, coefficients are shape factors. They say "for this
    aircraft at this angle, lift is CL times (pressure x area)". Multiply by
    real pressure and area later and you get real newtons.

        CL  lift coefficient   -- how good it is at making lift right now
        CD  drag coefficient   -- how much it is resisting the airflow
        Cm  pitching moment    -- nose-up/nose-down twist
        Cl  rolling moment     -- roll twist (note: lowercase l, not one)
        Cn  yawing moment      -- nose-left/nose-right twist

    Each of the three moments has the same two-part shape:

        (something) * angle        -> STIFFNESS: pushes back when disturbed
      + (something) * rotation rate -> DAMPING:   resists spinning, like a
                                       paddle dragging through water

    The (chord / 2V) and (span / 2V) factors just convert a rotation rate
    into an equivalent angle, so the two parts can be added together.

    "Clean" means the wing on its own -- the flaps are added separately in
    elevons.py so nothing gets counted twice.

    Computes the wing's force/moment coefficients (CL, CD, Cm, Cl, Cn) at
    the current angle of attack. These are "clean" -- they don't include
    the elevons, which are added separately (see elevons.py).

    CL/CD: uses a lookup table from real wind-tunnel/XFLR5 data if one is
    given (params.polar_table_path), and a simple backup formula outside
    the table's range so the simulator never runs out of data.

    Cm/Cl/Cn: always use the simple formula, even if a table is given --
    the table's Cm isn't reliable enough to mix in safely.
    """
    # Never divide by a speed of zero: floor it. Below this speed the damping
    # terms are meaningless anyway.
    Va_s = max(Va, params.Va_reg)                                       # safe speed for the "divide by V" damping terms
    CL, CD = _lift_drag_coefficients(alpha_wing, params)

    Cm = (params.Cm0                                                    # twist the wing has even at zero lift (from its sweep + twist)
          + params.Cm_alpha * alpha_wing                                # stiffness: more angle -> more nose-down push
          + params.Cm_q * (params.c_bar / (2.0 * Va_s)) * q_rate)       # damping: resists pitching, like air resistance on a paddle

    Cl = (params.Cl_beta * beta                                         # roll caused by being blown sideways
          + params.Cl_p * (params.b / (2.0 * Va_s)) * p_rate)           # damping: resists rolling

    Cn = (params.Cn_beta * beta                                         # weathercock: tries to point the nose into the wind
          + params.Cn_r * (params.b / (2.0 * Va_s)) * r_rate)           # damping: resists yawing

    return CL, CD, Cm, Cl, Cn


def _attached_CL_CD(alpha_wing, params):
    """
    The simple textbook wing formula, for when we have no measured data.

    In plain terms:
        lift  grows in a straight line with angle  (CL0 + slope * angle)
        drag  = a fixed minimum + a penalty proportional to lift SQUARED

    That second part is "induced drag" -- the unavoidable cost of making
    lift. Make twice the lift and you pay four times the induced-drag
    penalty. Short stubby wings (small AR) pay much more of it, which is why
    AR is in the denominator.

    Deliberately contains NO stall. Stall is applied ONCE, later, in
    _lift_drag_coefficients(), so it works identically whether the numbers
    came from here or from the measured table.

    ATTACHED-FLOW (pre-stall) lift and drag from the parametric model.

    Deliberately contains NO stall behaviour -- stall is applied once, in
    _lift_drag_coefficients(), so it is applied identically whether the
    attached data came from here or from the XFLR5 table. Previously the
    stall blend lived here, which meant it was skipped entirely whenever the
    table was in range (the table spans +/-23 deg and has no stall break of
    its own), so the simulator had no stall at all across its whole normal
    operating range.
    """
    CL = params.CL0 + params.CL_alpha * alpha_wing                       # straight-line lift growth with angle
    CD = params.CDp + (CL ** 2) / (np.pi * params.oswald_e * params.AR)  # minimum drag + the price of making lift
    return CL, CD


def _flat_plate_CL_CD(alpha_wing, params):
    """
    What the wing does once it is FULLY stalled: behave like a barn door.

    In plain terms: a stalled wing is no longer steering air smoothly. It is
    just a flat surface being shoved by the wind, with the force pointing
    roughly straight out of its face. Splitting that single force into
    "lift" and "drag" gives the two formulas below.

    Sanity check at 90 degrees (wing exactly side-on to the wind):
        sin(90)=1, cos(90)=0  ->  CL = 0     (no lift at all -- correct)
                                   CD = CN   (maximum drag -- correct)

    FULLY SEPARATED (post-stall) lift and drag.

    A stalled wing behaves like a flat plate: the resultant force is
    essentially normal to the surface with coefficient CN, resolved into
    wind axes as

        CL = CN sin(a) cos(a)      CD = CDp + CN sin^2(a)

    so CL -> 0 and CD -> CN at alpha = 90 deg, both correct. The old model
    blended CL this way but then fed the blended CL back into the INDUCED
    drag formula, so CD collapsed to CDp (0.02) at 90 deg instead of rising
    to ~1.2 -- a ~60x underestimate of drag in deep stall.
    """
    CN = _flat_plate_CN(params)                 # how draggy this shape is when fully stalled
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)
    CL = CN * sa * ca                      # sign follows sin(a), so it flips correctly for negative angles
    CD = params.CDp + CN * sa * sa         # grows to the full barn-door value at 90 degrees
    return CL, CD


# HOW THE MEASURED DATA AND THE FORMULA ARE COMBINED
#
# We have real XFLR5 numbers, but only over a limited range of angles
# (-23 to +24 degrees here). Outside that we have to fall back on the
# formula. Switching abruptly at the edge would put a kink in the physics,
# so instead _table_trust_weight() returns "how much do I trust the table
# right now" (1 inside, fading to 0 outside) and the two sources are mixed
# in that proportion. The result eases smoothly from real data to formula.
#
# The next two functions blend real measured data (the optional XFLR5
# table) with the backup formula above. The table only covers a limited
# angle-of-attack range, so _table_trust_weight works out how much to
# trust it at the current angle (1 = fully inside the table's range,
# 0 = fully outside it), and _lift_drag_coefficients uses that number to
# mix the table's CL/CD with the backup formula's CL/CD -- so instead of
# a sudden jump when the angle crosses the table's edge, the sim eases
# smoothly from "real data" to "backup formula".
def _table_trust_weight(alpha_wing, table, blend_width_rad):
    """
    How much should we trust the measured table at this angle? 1 = fully,
    0 = not at all.

    In plain terms: inside the range the data actually covers, trust it
    completely. Step outside and confidence fades away over the next few
    degrees, following a smooth S-curve rather than dropping off a cliff.

    1.0 anywhere INSIDE the table's own alpha range (real computed data
    is trusted fully, never discounted near its own edges), tapering by
    cosine ease down to 0.0 over blend_width_rad beyond whichever edge
    alpha_wing has crossed. Continuous at the boundary: right at the
    edge, weight=1 either way you approach it.
    """
    if table.alpha_min <= alpha_wing <= table.alpha_max:
        return 1.0   # we have real data here -- use it, no discount
    # How far outside the data have we strayed?
    d_outside = (table.alpha_min - alpha_wing) if alpha_wing < table.alpha_min \
        else (alpha_wing - table.alpha_max)   # distance past whichever edge
    if d_outside >= blend_width_rad:
        return 0.0   # far outside -- the formula is all we have
    t = 1.0 - d_outside / blend_width_rad  # 1 right at the edge, falling to 0 further out
    return 0.5 - 0.5 * np.cos(np.pi * t)   # smooth S-curve, no sudden jump


def _attached_from_best_source(alpha_wing, params):
    """
    Get pre-stall lift/drag from whichever source is most trustworthy here:
    measured data where we have it, formula where we don't, mixed smoothly
    in the overlap.

    Attached-flow CL/CD, taken from the measured table where it is trusted
    and from the parametric model elsewhere, with a smooth handover.
    """
    CL_param, CD_param = _attached_CL_CD(alpha_wing, params)   # the formula's answer, always available
    if not params.polar_table_path:
        return CL_param, CD_param                              # no table configured -> formula everywhere

    from . import polar_table
    table = polar_table.load_xflr5_polar(params.polar_table_path)
    weight = _table_trust_weight(alpha_wing, table, np.radians(params.polar_blend_deg))
    if weight <= 0.0:
        return CL_param, CD_param                              # too far outside the data
    CL_tab, CD_tab, _Cm_tab = table.interp(alpha_wing)   # Cm from the table is unused, see docstring
    if weight >= 1.0:
        return float(CL_tab), float(CD_tab)                    # squarely inside the data
    # In the overlap: weighted mix of the two.
    return (float(weight * CL_tab + (1.0 - weight) * CL_param),
            float(weight * CD_tab + (1.0 - weight) * CD_param))


def _lift_drag_coefficients(alpha_wing, params):
    """
    The final CL and CD, with stall and airframe drag included.

    Three things happen here, in order:

      1. Get the ATTACHED (flying normally) numbers -- measured or formula.
      2. Get the SEPARATED (fully stalled, barn-door) numbers.
      3. Mix them using sigma, which slides 0 -> 1 as the wing stalls.
         Below stall you get almost pure (1), well past it almost pure (2).
      4. Add the drag of everything that is NOT the wing.

    WHY STALL IS APPLIED HERE AND NOT EARLIER: the measured XFLR5 table has
    NO stall in it at all -- its lift just keeps rising all the way to +24
    degrees, because it is an inviscid result. If we simply trusted the table
    wherever it had data, the simulator would show a wing that never stalls
    across its entire normal operating range. Applying stall here, after the
    data source has been chosen, means it works the same either way.

    CL/CD with stall ALWAYS applied, whatever the attached data source.

    Two stages, in this order:

      1. attached-flow CL/CD  -- XFLR5 table where trusted, parametric outside
      2. blend into the fully separated flat-plate model past alpha_stall

    The second stage is the fix for a real defect: the XFLR5 polar supplied
    with this vehicle spans -23.25 to +24.00 deg and rises MONOTONICALLY
    across all of it (CL = +1.02 at 24 deg), i.e. it contains no stall at
    all -- it is a linear/inviscid result, not a viscous one. Because the
    old code returned raw table values whenever alpha was inside that range,
    the simulator produced attached, ever-increasing lift through and well
    beyond the 13 deg stall angle. Any transient that overshot the cap was
    therefore unphysical (observed: alpha_wing reaching -14.8 deg with a
    perfectly straight lift line through it).
    """
    CL_att, CD_att = _attached_from_best_source(alpha_wing, params)   # flying-normally values
    CL_sep, CD_sep = _flat_plate_CL_CD(alpha_wing, params)            # fully-stalled values
    sigma = stall_blend_sigma(alpha_wing, params.alpha_stall + _POST_STALL_OFFSET,
                               params.stall_blend_M)                  # 0 = attached, 1 = stalled
    CL = (1.0 - sigma) * CL_att + sigma * CL_sep                      # slide between the two
    CD = (1.0 - sigma) * CD_att + sigma * CD_sep

    # ---- Drag of everything that is NOT the wing -------------------------
    #
    # PLAIN VERSION: the wing is not the whole aircraft. The body, the four
    # motor pods, the booms and all the joins between them add drag too. The
    # XFLR5 table only ever knew about the wing, so that drag has to be added
    # here or the aircraft comes out unrealistically slippery.
    #
    # Airframe parasitic drag: fuselage, rotor nacelles, booms, arms, interference
    # -- everything that is NOT the lifting surface. Referenced to the wing area S
    # like every other coefficient here, and added regardless of data source.
    #
    # This matters more than it looks. The XFLR5 polar is a LIFTING-SURFACE result:
    # its viscous drag comes from the 2-D section polars along the wing strips, so
    # it accounts for the wing and nothing else. Whatever the table says at low
    # alpha (0.0064 here) is therefore the CLEAN WING, not the aircraft. Leaving
    # this at zero models a vehicle with four exposed rotors and a body that
    # somehow drags less than a sailplane.
    #
    # See params.CD_airframe -- flagged NEEDS_MEASUREMENT, since the honest value
    # comes from a drag buildup or a measurement rather than from this file.
    CD = CD + getattr(params, "CD_airframe", 0.0)
    return float(CL), float(CD)


def clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, q_rate, Va, beta, params, alpha_wing=0.0):
    """
    STEP 4: turn the dimensionless coefficients into REAL forces and moments.

    In plain terms, every line here is the same idea:

        real force = (how hard the air hits) x (how big the wing is) x (shape factor)
                   =        qbar_wing        x         S             x      C

    and for moments you additionally multiply by a lever arm (b for roll/yaw
    since those act across the span, c_bar for pitch since that acts along
    the chord).

    THE PITCHING MOMENT IS THE INTERESTING ONE. It has three parts:

      1. A built-in nose-up twist the wing has even at zero lift, which on
         this aircraft comes from its swept-back shape combined with the
         twist built into the wingtips.

      2. Lift acting through a LEVER ARM. The wing's lift does not act at the
         centre of gravity -- it acts 13 mm behind it. Because it is behind,
         more lift pushes the nose DOWN, which is what makes the aircraft
         naturally stable: if it pitches up, it makes more lift, which pushes
         it back down again.

      3. Damping, resisting whatever pitch rate it already has.

    Turns the coefficients into actual forces (Newtons) and moments
    (Newton-meters).

    My_aero (pitch moment) is built from the lift force acting at the
    neutral point, plus pitch damping -- it does NOT also add
    Cm_alpha*alpha separately, because that would double-count the same
    physical effect (the NP lever arm on Lclean already captures it).

    wing_scale fades the wing's entire force/moment contribution to zero
    as Va -> 0. At hover there's no real freestream, so alpha_wing is
    driven purely by the downwash cap (see downwash()) and can sit at a
    physically extreme value (e.g. -85 deg) with no real airflow behind
    it -- treating that as if it were genuine wind-tunnel lift/drag
    produces a small but constant, nonzero trim moment even at a perfect,
    zero-rate hover, which a normal attitude/rate controller reads as a
    disturbance to correct and ends up chasing into a slow, growing
    drift. A real wing does nothing useful with no airspeed, so this
    ramps its contribution in smoothly (full effect by Va=Va_reg) instead
    of pretending it's already flying.

    Below _WING_VA_DEADBAND, wing_scale is held at exactly 0.0 rather than
    just asymptotically approaching it. A plain min(Va/Va_reg, 1.0) still
    lets a genuinely negligible Va (a few mm/s of numerical residual, not
    real airspeed) multiply through qbar_wing -- which is large (rotor
    downwash pressure, not true dynamic pressure -- see downwash()) -- into
    a nonzero moment. That moment nudges the plant, which nudges Va
    further from zero, which produces slightly more moment: a slow
    self-feeding drift that never actually settles at true hover. A hard
    floor removes the feedback path entirely instead of merely shrinking it.
    """
    wing_scale = wing_activity(Va, params)   # 0 = parked, 1 = properly flying; multiplies EVERYTHING below
    Va_s = max(Va, params.Va_reg)            # safe speed for the damping divisions
    # pressure x area x shape factor = force. The CL_q part is extra lift generated
    # simply by the aircraft rotating in pitch.
    Lclean = wing_scale * qbar_wing * params.S * (CL + params.CL_q * (params.c_bar / (2.0*Va_s)) * q_rate)  # LIFT, newtons
    D = wing_scale * qbar_wing * params.S * CD                      # DRAG, newtons
    Y = wing_scale * qbar_wing * params.S * params.CY_beta * beta   # SIDEFORCE, newtons (only when sideslipping)

    # ---- The lever arm that makes the aircraft pitch-stable ---------------
    #
    # PLAIN VERSION: the wing's total push does not act at the centre of
    # gravity; it acts 13 mm BEHIND it. Pushing behind the balance point is
    # what makes the nose drop when lift increases -- the self-correcting
    # behaviour that keeps the aircraft the right way up.
    #
    # We use Fz (the push measured in the AIRCRAFT's own axes) rather than the
    # raw lift, because at 25-30 degrees nose-up those two differ noticeably.
    #
    # Pitching moment about the CG = (lever arm from CG to NP) x (aero force).
    # With stations measured aft from the nose, the NP sits at -(xnp - xcg) along
    # body +x (forward), so r x F leaves My = (xnp - xcg) * Fz.
    #
    # Fz is the BODY-axis normal force, not the wind-axis lift. Using Lclean
    # directly (as this did before) drops the cos(alpha) and the drag's own
    # lever contribution: exact at alpha=0, ~2% low at 11 deg, and materially
    # wrong once the vehicle is at the 25-30 deg body incidence these turn
    # cases actually fly at.
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)
    sb, cb = np.sin(beta), np.cos(beta)
    Fz_wing = -(Lclean * ca + D * sa * cb + Y * sa * sb)   # the wing's total push measured straight down through the aircraft

    Mx_aero = wing_scale * qbar_wing * params.S * params.b * Cl   # ROLL moment (span is the lever arm)
    # Cm0 here is the zero-lift pitching couple. On THIS aircraft it does not come
    # from camber -- every section is symmetric (NACA 0017/0017/0014/0012). It comes
    # from -2 deg of tip WASHOUT acting on a 28.81 deg swept planform: at zero net
    # lift the washed-out tips carry download and, being swept aft, that download
    # sits behind the CG and pitches the nose up. This is how a tailless swept wing
    # gets its pitch-up couple without a reflexed section. Measured from the polar
    # as +0.00196; at zero lift the moment is a pure couple, so that figure is
    # independent of whatever reference point XFLR5 used for its Cm column.
    My_aero = (wing_scale * qbar_wing * params.S * params.c_bar * params.Cm0        # built-in nose-up twist (sweep + wingtip washout)
               + (params.xnp - params.xcg) * Fz_wing                                # lift acting behind the CG -> the stabilising nose-down push
               + wing_scale * qbar_wing * params.S * params.c_bar
                 * params.Cm_q * (params.c_bar / (2.0 * Va_s)) * q_rate)            # resistance to pitching
    Mz_aero = wing_scale * qbar_wing * params.S * params.b * Cn   # YAW moment (span is the lever arm)

    # WHY Cm_alpha IS MISSING HERE: it describes exactly the same physical
    # effect as the lever-arm term above (lift acting behind the CG). Including
    # both would count that effect twice and make the aircraft look twice as
    # pitch-stable as it really is. Same story for yaw and Cn_beta.
    #
    # NOTE: Cm_alpha is deliberately NOT added -- the NP lever arm above already
    # IS the alpha-dependent pitching moment, and adding both double-counts it.
    # Same reasoning for yaw: Cn_beta carries the whole weathercock effect, so no
    # separate sideforce-times-lever-arm term is added to Mz_aero either.
    return Lclean, D, Y, np.array([Mx_aero, My_aero, Mz_aero])


def project_to_body(Lclean, D, Y, Moments, alpha_wing, beta):
    """
    STEP 5: convert from "relative to the airflow" into "relative to the
    aircraft".

    In plain terms: lift and drag are defined against the AIRFLOW -- drag
    points straight back along the wind, lift points straight up across it.
    But the physics engine needs forces in the AIRCRAFT's own directions
    (out the nose, out the wing, through the floor). Since the airflow
    arrives at angle alpha (and beta sideways), we rotate by those angles.

    Sanity check at alpha = 0 (air coming straight down the nose):
        sin(0)=0, cos(0)=1  ->  Fx = -D   (drag pushes straight back)
                                Fz = -L   (lift pushes straight up; up is
                                           negative z because z points down)

    THE MOMENTS ARE NOT ROTATED. That is deliberate, not an oversight -- see
    the note below.

    Rotates the wind-axis forces (L, D, Y) into the body-axis forces
    (Fx, Fy, Fz) the rest of the sim uses, accounting for both alpha
    and sideslip (beta):

        Fx = L sin(a) - D cos(a) cos(b) - Y cos(a) sin(b)
        Fy = Y cos(b) - D sin(b)
        Fz = -L cos(a) - D sin(a) cos(b) - Y sin(a) sin(b)
    """
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)   # rotation for angle of attack
    sb, cb = np.sin(beta), np.cos(beta)               # rotation for sideslip
    Fx = Lclean * sa - D * ca * cb - Y * ca * sb      # force out the nose
    Fy = Y * cb - D * sb                              # force out the right wing
    Fz = -Lclean * ca - D * sa * cb - Y * sa * sb     # force down through the floor (so lift is negative)

    # WHY THE MOMENTS PASS STRAIGHT THROUGH
    #
    # PLAIN VERSION: lift and drag genuinely need rotating, because they are
    # defined against the wind. The moments do NOT -- they were already worked
    # out in the aircraft's own axes (Cl_p multiplies the aircraft's own roll
    # rate, the lever arm is a distance along the aircraft's own nose-to-tail
    # line). Rotating them a second time would be applying the same correction
    # twice, which used to leak roll into yaw.
    #
    # Moments are passed through UNROTATED. L, D and Y are wind-axis forces and
    # genuinely need the wind-to-body rotation above, but Cl/Cm/Cn -- and the NP
    # lever-arm term built from them in clean_forces_moments() -- are already
    # BODY-axis quantities by definition: Cl_p multiplies body roll rate p,
    # Cn_r multiplies body yaw rate r, and the lever arm is a body-x distance.
    # Rotating them a second time mixed roll into yaw by sin(alpha) -- 47% cross
    # contamination at the 28 deg body incidence of the 8g turn case. It stayed
    # invisible in the validated cases only because those all run at beta=0 with
    # zero roll/yaw rate, which makes Cl = Cn = 0.
    return np.array([Fx, Fy, Fz]), np.asarray(Moments, dtype=float).copy()
