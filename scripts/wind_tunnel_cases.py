"""
Named test cases for the wind tunnel GUI. Each case just states the FLIGHT
CONDITION (airspeed, flight-path angle, and/or turn load factor) -- never
actuator values. The GUI's case solver works out the tilt, elevon, and
rotor speeds that actually satisfy that condition, automatically.

To add a case: add an entry to CASES below. `n_load` is the turn/maneuver
load factor (1.0 = plain climbing/level flight, no turn; >1.0 = a
coordinated turn requiring that many g's of lift). `gamma_deg` is the
flight-path angle (0 = level). `hover` = True skips solving entirely and
uses the known exact hover trim. `interactive` = True means this case
isn't solved/run here at all -- selecting it and clicking "Solve Case &
Run" launches the separate live slider tool (wind_tunnel_live.py) instead.
"""

CASES = {
    "1: Hover": dict(
        hover=True,
        description="Steady hover, no wind, no maneuvering.",
    ),
    "2: Head-on interception (50 m/s)": dict(
        hover=False,
        V=50.0,
        gamma_deg=4.574,   # arctan(H/D) = arctan(200/2500), common line-of-sight climb geometry
        n_load=1.0,        # straight climbing flight, no turn
        description="D=2500m, H=200m, d_slant=2508m. Climbing straight toward the target line of sight.",
    ),
    "3: Coordinated turn, 5.2g (R=50m)": dict(
        hover=False,
        turn_radius_m=50.0,   # V is DERIVED from this + n_load, never hand-entered
        gamma_deg=0.0,        # level turn
        n_load=5.2,
        description="R=50m turn radius, bank angle ~78.9deg (derived from n_load). "
                     "Wing flown 2deg below the stall cap, pitch trimmed by front/aft "
                     "rotor speed split with elevons neutral.",
    ),
    "4: Interactive Hover (live sliders)": dict(
        interactive=True,
        description="Starts at true hover (wind=0, tilt=0, elevons=0). Live sliders for "
                     "both elevons, rotor tilt, and wind velocity -- opens a separate "
                     "window (wind_tunnel_live.py).",
    ),
    "5: Coordinated turn, 8g (R=32.2m)": dict(
        hover=False,
        turn_radius_m=32.2,   # V is DERIVED from this + n_load, never hand-entered
        gamma_deg=0.0,        # level turn
        n_load=8.0,
        description="R=32.2m turn radius, bank angle ~82.8deg. Same 50 m/s airspeed as "
                     "Case 3 -- the tighter radius is what buys the extra g. Peak rotor "
                     "demand roughly doubles versus 5.2g.",
    ),
}
