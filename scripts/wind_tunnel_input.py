"""
Wind tunnel test inputs -- edit these, then hit Run in the GUI window. The
GUI re-reads this file fresh every time you click a button, so you can edit
and re-run without restarting it. Note: "Solve Trim" and "Solve Case" both
OVERWRITE the rotor speed / tilt / elevon / wind values below with their
solved results before running -- manual edits here only matter for "Run
(file as-is)", or right after a solve if you want to nudge the result by
hand before the next run.

Rotor order is [FR, AL, FL, AR] everywhere below -- Front-Right, Aft-Left,
Front-Left, Aft-Right -- matching the vehicle's actual rotor layout and spin
directions (see src/models/InterceptorAllocator.py's docstring). This is the
same order the physics engine itself uses, so there's no relabeling between
what you type here and what the vehicle actually does.
"""

# ------------------------------ Rotor speeds (rev/s) ------------------------------
ROTOR_SPEED_FR = 386.629710
ROTOR_SPEED_AL = 262.645108
ROTOR_SPEED_FL = 386.629710
ROTOR_SPEED_AR = 262.645108

# ------------------------------ Rotor tilt angles (deg) ------------------------------
# 0 = hover (thrust straight up), 90 = full cruise (thrust straight forward)
ROTOR_TILT_FR = 73.219770
ROTOR_TILT_AL = 73.219770
ROTOR_TILT_FL = 73.219770
ROTOR_TILT_AR = 73.219770

# ------------------------------ Elevon deflection (deg) ------------------------------
# Independent channels. Equal values (same sign) = pure symmetric/pitch input, no roll.
# Opposite signs = pure roll input, no pitch. Any mix of the two = combined pitch+roll.
ELEVON_1_DEG = 0.000000
ELEVON_2_DEG = 0.000000

# ------------------------------ Wind / test conditions ------------------------------
WIND_VELOCITY_MPS = 50.029913
RUN_DURATION_S = 3.0
CLAMPED_BALL_SOCKET = True  # True: position frozen, free rotation only (pitch/roll/yaw/bank).
                             # False: free 6DOF flight (can also translate).

# ------------------------------ Solve Trim settings ------------------------------
# Only used when you click "Solve Trim" -- solves for the alpha, elevon deflection, and
# rotor TILT that balance the vehicle at WIND_VELOCITY_MPS and this flight-path angle,
# given the rotor SPEEDS above (front/aft averaged). Requires WIND_VELOCITY_MPS > 0.
TARGET_FLIGHT_PATH_ANGLE_DEG = 4.574
