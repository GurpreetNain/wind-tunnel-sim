"""
ELEVONS -- the two moving flaps on the back edge of the wing.

PLAIN-ENGLISH OVERVIEW
----------------------
A normal aeroplane has separate elevators (pitch) and ailerons (roll). A
flying wing has no tail, so it uses two combined surfaces called ELEVONS --
one on each wing's trailing edge -- and gets both jobs out of them:

    both flaps move the SAME way   -> the nose pitches up or down
    the flaps move OPPOSITE ways   -> the aircraft rolls

So this file's first job is to take the two physical flap angles the pilot
(or the solver) commanded and split them into "how much pitch" and "how much
roll". Everything after that turns those two numbers into actual forces in
newtons and moments in newton-metres.

Three real-world effects are modelled on top of the simple version:

  1. BIG DEFLECTIONS WASTE THEMSELVES. Doubling the flap angle does not
     double the effect -- past roughly 15-20 degrees the air starts to
     separate off the flap and you get diminishing returns. That is the
     "efficiency" factor eta.

  2. A DEFLECTED FLAP ADDS DRAG. Sticking a plate into the airflow slows you
     down, like a small airbrake. That is the dDe term.

  3. FLAPS STOP WORKING WHEN THE WING STALLS. A flap only works if the air
     is still flowing smoothly over the wing in front of it. Once the wing
     stalls, the flap is sitting in dead, churning air and most of its power
     disappears -- exactly when you would most want it back.

Everything here is evaluated in qbar_wing and alpha_wing (the values AFTER
the rotor wash has been accounted for), not the plain freestream values,
because the elevons physically sit inside the rotor slipstream.
"""
import numpy as np # Import the numpy library for math operations

# How much of the elevon's power survives once the wing is fully stalled.
# 0.15 means "about 15% left". Not zero -- a deflected plate sitting in
# churning air still pushes a little -- but nowhere near its normal strength.
_POST_STALL_EFFECTIVENESS = 0.15


def split(delta1, delta2):
    """
    Turn the TWO physical flap angles into the TWO things we actually care
    about: pitch demand and roll demand.

    In plain terms:
      - If both flaps move the same way, they cancel out in the roll sense
        and add up in the pitch sense  -> pure pitch.
      - If they move opposite ways, they cancel in pitch and add in roll
        -> pure roll.

    Worked example: delta1 = +10 deg, delta2 = +10 deg (both up together)
        delta_e = (10 + 10)/2 = +10 deg   <- full pitch command
        delta_a = (10 - 10)/2 =   0 deg   <- no roll at all

    Another: delta1 = +10 deg, delta2 = -10 deg (opposite ways)
        delta_e = (10 - 10)/2 =   0 deg   <- no pitch
        delta_a = (-10 - 10)/2 = -10 deg  <- full roll command

    delta_e (pitch, symmetric), delta_a (roll, differential).
    """
    delta_e = 0.5 * (delta1 + delta2) # AVERAGE of the two = the part that pitches
    delta_a = 0.5 * (delta2 - delta1) # DIFFERENCE of the two = the part that rolls
    return delta_e, delta_a           # Return the calculated pitch and roll inputs


def efficiency(delta_rad, params):
    """
    How much of a flap deflection actually "counts".

    In plain terms: a flap is not twice as effective when you deflect it
    twice as far. At small angles the air follows the flap neatly and you
    get nearly the full theoretical effect. At large angles the airflow
    starts peeling away from the flap surface and a good chunk of the
    deflection is wasted.

    This returns a number between roughly 0 and 1 -- think of it as
    "fraction of the textbook effect you really get" -- looked up from a
    DATCOM table of measured values. 1.0 = full effect, 0.5 = half wasted.

    Only the SIZE of the deflection matters, not its direction, which is why
    the absolute value is taken before the lookup.

    DATCOM large-deflection efficiency eta(delta)
    """
    deg = np.degrees(np.abs(delta_rad)) # Convert the absolute deflection angle from radians to degrees
    return float(np.interp(deg, params.eta_table_deg, params.eta_table_val)) # Look up and interpolate the efficiency factor from the DATCOM table


def apply_stall_constraint(delta_e, alpha_wing, params):
    """
    A safety rule: once the wing is already at its stall limit, do not allow
    a flap deflection that would make the stall worse.

    In plain terms: the wing stalls when it is tilted too steeply into the
    airflow. Deflecting the elevon DOWNWARD (negative here) effectively adds
    even more curvature to the back of the wing, which is like asking the
    already-struggling airflow to turn even harder. So at or past the stall
    angle we simply forbid downward deflection -- upward is still fine,
    because that unloads the wing and helps.

    This is a hard operating limit (an if-statement), not a smooth physical
    effect. The smooth loss-of-authority effect is separate and lives in
    elevon_forces_moments() below.

    delta_e >= 0 once the wing is at/above the stall cap: down-elevon
    would push the near-stall aft wing past separation.
    Doc4 Sec 11.6; FINAL_CASES Sec 2 ("Stall constraint").
    Hard operating limit enforced here, not part of the continuous
    dynamics -- mirrors how it is used in the trim cases ("Wing at cap: delta_e = 0").
    """
    if alpha_wing >= params.alpha_stall:    # Is the wing already at or past its stall angle?
        return max(delta_e, 0.0)            # Yes -> allow up-elevon only; clamp anything negative to 0
    return delta_e                          # No -> the requested deflection is fine as-is


def elevon_forces_moments(delta_e, delta_a, alpha_wing, qbar_wing, params):
    """
    Turn the pitch and roll commands into real forces (newtons) and moments
    (newton-metres).

    In plain terms, four things happen when you deflect the elevons:

      1. LIFT CHANGES.   Deflecting the flaps up pushes the tail of the wing
                         down, which pitches the nose UP -- but it does that
                         by REMOVING some lift. That is why dL is negative
                         for a positive (up) deflection. It is the price you
                         pay for pitch control on a tailless aircraft.

      2. DRAG INCREASES. The deflected flap acts like a small airbrake. Note
                         it depends on the SQUARE of the deflection, so it is
                         negligible at small angles and grows fast at large
                         ones.

      3. PITCH MOMENT.   The actual nose-up/nose-down twisting force.

      4. ROLL MOMENT.    Only if the two flaps disagree (delta_a not zero).
                         One wing gains lift, the other loses it, and the
                         aircraft rolls. Mz is zero because this aircraft has
                         no rudder or fin -- there is nothing to make yaw with.

    Note everything uses qbar_wing and alpha_wing (post-rotor-wash values),
    because the elevons sit in the rotor slipstream, not the clean freestream.

        Delta_L   = -CL_delta_e * eta(delta_e) * delta_e * qbar_wing * S
        Delta_De  =  qbar_wing * S * k_e * (eta(delta_e)*delta_e)^2
        Fx_delta  =  Delta_L*sin(a) - Delta_De*cos(a)
        Fz_delta  = -Delta_L*cos(a) - Delta_De*sin(a)
        Mx_delta  =  2*qbar_wing*S*CL_delta_e*eta(delta_a)*delta_a*y_elevon
        My_delta  =  Cm_delta_e*eta(delta_e)*delta_e*qbar_wing*S*c_bar
        Mz_delta  =  0   (no rudder; finless airframe)
    """
    eta_e = efficiency(delta_e, params) # How much of the PITCH deflection actually counts (0..1)
    eta_a = efficiency(delta_a, params) # How much of the ROLL deflection actually counts (0..1)

    # ---- Second efficiency knockdown: is the wing in front of the flap stalled?
    #
    # PLAIN VERSION: a flap can only steer air that is still flowing smoothly
    # past it. Once the wing stalls, the flap is sitting in a churning wake and
    # barely does anything. sigma below is a number from 0 (wing flying normally)
    # to 1 (wing fully stalled) -- the SAME sigma the wing model itself uses, so
    # the two files always agree about whether the wing has stalled.
    #
    # authority then slides from 1.0 (full control power) down to 0.15 (almost
    # none) as the stall develops.
    #
    # Stall knockdown. A trailing-edge surface only works while the flow over the
    # wing ahead of it is still attached -- once the wing separates, the elevon
    # sits in dead air and most of its authority goes with it. Without this the
    # model hands the controller full pitch/roll power at any angle of attack,
    # which is exactly backwards: authority is supposed to VANISH right when a
    # stall recovery needs it. Faded to _POST_STALL_EFFECTIVENESS rather than to
    # zero, since a deflected surface in separated flow still does a little.
    from .lifting_body import stall_blend_sigma, _POST_STALL_OFFSET
    sigma = stall_blend_sigma(alpha_wing, params.alpha_stall + _POST_STALL_OFFSET,
                              params.stall_blend_M)
    authority = 1.0 - (1.0 - _POST_STALL_EFFECTIVENESS) * sigma   # 1.0 normally -> 0.15 fully stalled
    eta_e *= authority   # pitch power fades as the wing stalls
    eta_a *= authority   # roll power fades the same way

    # ---- The two aerodynamic effects, in newtons -------------------------
    # dL is NEGATIVE for an up (positive) deflection: you buy nose-up pitch by
    # giving away lift. qbar_wing * S converts a coefficient into a real force.
    dL = -params.CL_delta_e * eta_e * delta_e * qbar_wing * params.S   # lift LOST because the flaps are deflected
    # Squared, so it is tiny at small angles and grows quickly at large ones.
    dDe = qbar_wing * params.S * params.k_e * (eta_e * delta_e) ** 2   # extra drag, like a small airbrake

    # ---- Rotate those two into the aircraft's own axes -------------------
    # dL and dDe are defined relative to the AIRFLOW direction. The rest of the
    # simulator works in the aircraft's own frame (x = out the nose, z = down
    # through the floor), and the airflow arrives at angle alpha_wing, so we
    # rotate by that angle to convert between the two.
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)          # sin/cos of the angle between airflow and the aircraft
    Fx = dL * sa - dDe * ca                                  # force along the nose direction (mostly the drag pushing back)
    Fz = -dL * ca - dDe * sa                                 # force downward through the floor (mostly the lift change)

    # ---- The twisting moments, in newton-metres --------------------------
    # ROLL: the factor of 2 is because BOTH wings contribute -- one gains lift
    # while the other loses it, and both push the same way round. y_elevon is
    # how far out along the wing the flap's force effectively acts (the longer
    # the lever, the more roll you get for the same force).
    Mx = 2.0 * qbar_wing * params.S * params.CL_delta_e * eta_a * delta_a * params.y_elevon   # roll moment (only if the flaps disagree)
    # PITCH: c_bar (the average wing chord) is the lever arm for pitch.
    My = params.Cm_delta_e * eta_e * delta_e * qbar_wing * params.S * params.c_bar            # pitch moment (nose up / nose down)
    Mz = 0.0   # no yaw: this aircraft has no rudder and no fin

    Force = np.array([Fx, 0, Fz])    # Fy = 0: elevons cannot push the aircraft sideways
    Moment = np.array([Mx, My, Mz])


    return Force, Moment
    # return {"delta_L": dL, "delta_De": dDe, "Fx": Fx, "Fy": 0.0, "Fz": Fz,      # Pack the local forces (L, D) and body-axis forces (X, Y, Z) into a dictionary
    #         "Mx": Mx, "My": My, "Mz": Mz, "eta_e": eta_e, "eta_a": eta_a}       # Pack body-axis moments (X, Y, Z) and efficiencies into the same dictionary to return
