"""
Central parameter set for the winged quad-tiltrotor interceptor.

"""
from dataclasses import dataclass, field
import json
import os
import numpy as np


def _default_polar_table_path():
    """Absolute path to the real XFLR5 fuselage+wing polar. Computed from
    this file's own location (not a relative string) so it resolves the
    same way regardless of which directory a script is launched from --
    deltav scripts are normally run from src/models/, three levels above
    the project root where config/ lives."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(project_root, "config", "aero", "wing_polar_xflr5.txt")


@dataclass
class InterceptorParams:
    # ---- environment ----------------------------------------------------
    rho: float = 1.225                      # kg/m^3, sea level (Appendix C.2)
    g: float = 9.81                         # m/s^2

    # ---- mass & geometry (FINAL_CASES Sec 1, Appendix C.2) --------------
    mass: float = 3.0                       # kg
    S: float = 0.156                        # m^2, wing reference area
    b: float = 0.60                         # m, span
    c_bar: float = 0.291                    # m, MAC -- see chord note above
    AR: float = 2.30

    xcg: float = 0.198                      # m, from nose (selected CG, FINAL_CASES Sec 4)
    xnp: float = 0.211                      # m, from nose
    x_front_pivot: float = 0.034            # m, from nose
    x_aft_pivot: float = 0.330              # m, from nose
    l_front: float = 0.164                  # m = xcg - x_front_pivot
    l_aft: float = 0.132                    # m = x_aft_pivot - xcg

    # Rotor hub geometry -- H-frame layout confirmed by CAD/top-view sketch:
    # front pair 164 mm ahead of CG, aft pair 132 mm behind (matches
    # l_front/l_aft above); front pair spans +/-218.5 mm, aft pair spans
    # +/-451.5 mm -- the front and aft crossbars are DIFFERENT widths (an
    # H-shape, not a simple rectangular frame), so a single symmetric
    # "rotor_lateral_arm" cannot represent this -- replaced with explicit
    # per-rotor y stations below.
    rotor_y_front: float = 0.2185           # m, +/- half-span of the FRONT crossbar (rotors 1, 3)
    rotor_y_aft: float = 0.4515             # m, +/- half-span of the AFT crossbar (rotors 2, 4)

    y_elevon: float = 0.110                 # m, elevon lateral centroid arm (FINAL_CASES/App. C)
    x_elevon_chordwise: float = 0.268       # m, chordwise centroid from LE (informational only;
                                             # NOT used in the body-axis EOM, which take moment
                                             # arms as xcg/xnp/l_front/l_aft/y_elevon directly)
    max_thrust_per_motor: float = 45.0
    # ---- inertia tensor ---------------------------------------------------
    # NEEDS_MEASUREMENT: Appendix C.3 lists Ixx, Iyy, Izz, Ixz with NO
    # numeric value ("typical" placeholder only). Values below are a
    # flat-plate estimate using the known span (0.60 m) and an assumed
    # ~0.36-0.40 m fuselage length (x_aft_pivot - x_front_pivot region):
    #   Ixx (roll)  ~ m*b^2/12   -- resists the SPANWISE mass distribution
    #   Iyy (pitch) ~ m*L^2/12   -- resists the shorter FORE-AFT distribution
    #   Izz (yaw)   ~ Ixx + Iyy  (thin-body/perpendicular-axis approx.)
    # Since span > fuselage length here, Ixx > Iyy is the physically
    # correct ordering -- an earlier draft of this file had them
    # backwards (Ixx < Iyy), which was verified to produce a spuriously
    # fast (~20-40 rad/s growth rate) open-loop pitch instability in a
    # hover integration test. These are still coarse guesses and MUST be
    # replaced with a real mass-property estimate (CAD inertia or
    # bifilar/trifilar swing test) before any trim/control result is
    # treated as final -- real hardware (motors, battery, structure)
    # concentrated away from the CG typically increases these further.
    Ixx: float = 0.045                      # kg m^2   NEEDS_MEASUREMENT
    Iyy: float = 0.020                      # kg m^2   NEEDS_MEASUREMENT
    Izz: float = 0.060                      # kg m^2   NEEDS_MEASUREMENT
    Ixz: float = 0.003                      # kg m^2   NEEDS_MEASUREMENT

    # ---- lift / drag / moment coefficients ---------------------------------
    CL0: float = 0.0                        # symmetric section 
    CL_alpha: float = 5.0                   # /rad, "typical" at AR=2.3   NEEDS_MEASUREMENT
    CL_q: float = 3.0                       # /rad, Appendix C.3 "typical"              NEEDS_MEASUREMENT
    CDp: float = 0.02                       # WING profile drag, parametric model only.
                                             # NOTE: bypassed entirely whenever the XFLR5
                                             # table is in range (i.e. -23..+24 deg, the whole
                                             # normal envelope) -- the table supplies its own
                                             # wing CD.                    NEEDS_MEASUREMENT
    CD_airframe: float = 0.0                # Parasitic drag of everything that is NOT the
                                             # wing -- fuselage, nacelles, booms, interference
                                             # -- referenced to wing area S, added on top of
                                             # whatever CD the wing model returns.
                                             # 0.0 means "not yet accounted for", which
                                             # UNDERSTATES total drag and therefore the thrust
                                             # and tilt any trim needs. Hand analysis for this
                                             # airframe implies a total CD near 0.085 against
                                             # the wing's own ~0.010, i.e. CD_airframe ~ 0.075.
                                             #                             NEEDS_MEASUREMENT
    oswald_e: float = 0.8                   #  "typical"                    NEEDS_MEASUREMENT

    Cm0: float = 0.0                        #  "~0"
    Cm_alpha: float = None                  # computed in __post_init__ from the static-margin
                                             # identity Cm_alpha = -CL_alpha*(xnp-xcg)/c_bar
                                             # unless overridden explicitly
    Cm_q: float = -10.0                     # /rad,             NEEDS_MEASUREMENT

    # NEEDS_MEASUREMENT: lateral-directional derivatives are explicitly
    # left blank ("--") or only qualitatively described ("small","weak without a tail") 
    CY_beta: float = -0.10                  # /rad   NEEDS_MEASUREMENT
    Cl_beta: float = -0.05                  # /rad   NEEDS_MEASUREMENT
    Cl_p: float = -0.40                     # /rad   NEEDS_MEASUREMENT
    Cn_beta: float = 0.02                   # /rad, small & weakly stabilising   NEEDS_MEASUREMENT
    Cn_r: float = -0.05                     # /rad, weak, no tail                NEEDS_MEASUREMENT

    alpha_stall: float = np.radians(13.0)   # FINAL_CASES Sec 1 -- the AoA CAP the vehicle
                                             # is operated within (Cases 1-6); kept separate
                                             # from the polar table's own coverage below.
    stall_blend_M: float = 50.0             # Beard-McLain blend sharpness -- not given
                                             # numerically anywhere in the sources; 50 is the
                                             # standard literature default. NEEDS_TUNING --
                                             # only used when polar_table_path is None
                                             # (pure parametric fallback model).

    # --- XFLR5 polar table (replaces the parametric CL(alpha)/CD(alpha)
    # model within the table's own coverage) ------------------------------
    # Set to None to use the pure parametric stall-blend model everywhere
    # (the original behavior). When set, lifting_body.py trusts the table
    # for alpha inside its range and blends smoothly into the SAME
    # flat-plate extension used before (stall_blend_sigma) for alpha
    # outside the table -- e.g. a typical XFLR5 sweep only covers about
    # -23 to +24 deg, but hover's downwash-corrected alpha_wing routinely
    # reaches 80-90 deg, so the blend-to-flat-plate fallback is mandatory,
    # not optional, or the simulator has no defined aero data out there.
    polar_table_path: str = field(default_factory=_default_polar_table_path)
                                             # real fuselage+wing XFLR5 sweep (50 m/s,
                                             # -23.25 to 24 deg), config/aero/wing_polar_xflr5.txt.
                                             # Set to None to force the pure parametric
                                             # stall-blend model everywhere instead.
    polar_blend_deg: float = 5.0            # width of the cosine taper straddling each
                                             # edge of the table's alpha range, blending
                                             # table values -> flat-plate values. Not a
                                             # source concept -- an engineering choice,
                                             # same category as Va_reg/n_reg. NEEDS_TUNING

    # ---- elevon -----------------------------------------------------------
    CL_delta_e: float = 0.246               # /rad, XFLR5-calibrated (FINAL_CASES Sec 2 eq. 3)
    Cm_delta_e: float = 0.059               # /rad, at xcg = 198 mm (FINAL_CASES Sec 2 / Sec 5)
    k_e: float = 0.5                        # elevon drag factor, Appendix C.3 "typical"  NEEDS_MEASUREMENT
    eta_table_deg: tuple = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    eta_table_val: tuple = (1.00, 1.00, 0.87, 0.75, 0.66, 0.60)   # DATCOM, Doc4 Sec 11.5
    delta_max: float = np.radians(25.0)     # mechanical travel limit

    # ---- rotor / propulsion -------------------------------------------------
    # Real APC 10x7 propeller bench data (PER3_10x7.txt, RPM=10000 row --
    # picked because it's fully populated across the whole J range and mid-
    # range enough not to be a Reynolds-number outlier; the different RPM
    # sweeps in that file agree with each other to a few percent, so any one
    # of them would do). Rotor diameter MUST match the prop this table came
    # from -- CT/CQ are only meaningful nondimensionalized by the same D
    # they were measured with. 10 in = 0.254 m, not the old 8 in placeholder.
    rotor_diameter: float = 0.254           # m (10 in, APC 10x7)

    # CT(J), CQ(J) as interpolated tables over advance ratio J = V_N/(nD).
    # CQ here is derived from the file's power coefficient via the standard
    # propeller identity Cq = Cp / (2*pi), since P = Q * omega = Q*2*pi*n
    # implies Cp*rho*n^3*D^5 = Cq*2*pi*rho*n^3*D^5. Ip below is still a
    # placeholder -- the bench file gives no polar-inertia figure.
    CT_table_J:  tuple = (0.0000, 0.0303, 0.0606, 0.0908, 0.1211, 0.1514, 0.1817, 0.2119, 0.2422, 0.2725,
                          0.3028, 0.3331, 0.3633, 0.3936, 0.4239, 0.4542, 0.4844, 0.5147, 0.5450, 0.5753,
                          0.6056, 0.6358, 0.6661, 0.6964, 0.7267, 0.7570, 0.7872, 0.8175, 0.8478, 0.8781)
    CT_table_val: tuple = (0.1250, 0.1236, 0.1220, 0.1203, 0.1184, 0.1163, 0.1139, 0.1114, 0.1086, 0.1055,
                           0.1022, 0.0986, 0.0947, 0.0905, 0.0860, 0.0812, 0.0762, 0.0709, 0.0655, 0.0600,
                           0.0543, 0.0485, 0.0426, 0.0366, 0.0305, 0.0244, 0.0183, 0.0122, 0.0061, 0.0000)
    CQ_table_J:  tuple = CT_table_J
    CQ_table_val: tuple = (0.00824, 0.00840, 0.00856, 0.00871, 0.00885, 0.00898, 0.00909, 0.00918, 0.00925, 0.00929,
                           0.00931, 0.00928, 0.00922, 0.00910, 0.00894, 0.00874, 0.00850, 0.00820, 0.00785, 0.00745,
                           0.00702, 0.00654, 0.00602, 0.00546, 0.00484, 0.00420, 0.00352, 0.00279, 0.00204, 0.00124)
    n_reg: float = 5.0                      # rev/s floor for the J = V_N/(nD)
                                             # denominator -- same regularization
                                             # pattern as Va_reg, prevents J from
                                             # diverging at low commanded rpm.
                                             # NEEDS_MEASUREMENT / ENGINEERING CHOICE
    Ip: float = 2.0e-5                      # kg m^2, rotor polar inertia    NEEDS_MEASUREMENT
    k_eps: float = 0.75                     # downwash angle factor, range 0.5-1.0   NEEDS_TUNING
    k_q: float = 0.75                       # dyn. pressure augmentation, range 0.5-1.0             NEEDS_TUNING

    # ---- oblique-flow (tilted-disc) correction -----------------------------
    # Normal-force / P-factor terms (Rotor_Block_Oblique_Flow_Extension.pdf
    # Sec 7) that the axial CT(J)/CQ(J) table cannot see -- only triggered by
    # disc incidence alpha_i != 0 (in-plane flow at the hub), not by tilt
    # angle alone (Sec 5). Needs blade planform, not just the integrated
    # CT/CQ the PER3 file provides:
    blade_Nb: int = 2                       # blade count, APC-class 10x7   ASSUMED (standard 2-blade)
    blade_chord: float = 0.020              # m, representative mean chord (Sec 7.1)   NEEDS_MEASUREMENT
                                             # -- real value needs the APC .PEO geometry
                                             # file or digitized blade planform.
    blade_Cd: float = 0.02                  # mean blade section profile drag (Sec 7.2)  NEEDS_MEASUREMENT
                                             # -- representative for a thin low-Re section;
                                             # enters as a small additive term, not dominant.
    J_reg: float = 0.05                     # floor on |J| used ONLY inside the oblique
                                             # correction's pi/J singular terms (Sec 7.5:
                                             # "Eqs. 4-5 blow up as Ji -> 0 ... cannot be
                                             # reliably applied near hover"). Does NOT touch
                                             # the baseline CT/CQ table lookup (advance_ratio()
                                             # has its own n_reg floor for that). Same
                                             # regularization pattern as n_reg/Va_reg.
                                             # NEEDS_TUNING / ENGINEERING CHOICE

    # NOT specified anywhere in the source documents. The rate-damping
    # terms (CLq*c_bar/(2Va)*q, Cmq*c_bar/(2Va)*q, Clp*b/(2Va)*p,
    # Cnr*b/(2Va)*r) are classical non-dimensional stability derivatives
    # that are only meaningful for Va away from zero; none of the three
    # PDFs discuss their Va->0 (hover) limit. A naive 1e-3 m/s floor
    # still amplifies a tiny hover body-rate by a factor of ~150 and was
    # verified to produce a runaway pitch moment in an open-loop hover
    # integration test. Va_reg is the floor used in EVERY 1/Va rate-term
    # in lifting_body.py (not in Va itself, which is left un-floored for
    # airdata/downwash). Below this speed, rate damping smoothly stops
    # growing rather than diverging -- physically defensible (a hovering
    # vehicle has no meaningful "forward-flight" rate derivative) but is
    # an implementation choice beyond what the sources specify, and
    # should be revisited once real low-speed/hover rate-damping data
    # (e.g. from a whirl-tower or CFD sweep) is available.
    Va_reg: float = 2.0                     # m/s   NEEDS_MEASUREMENT / ENGINEERING CHOICE

    rotor_tilt_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    rotor_x_arm: np.ndarray = field(
        default_factory=lambda: np.array([0.164, -0.132, 0.164, -0.132]))
    rotor_spin_sign: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, -1.0, -1.0]))

    def __post_init__(self):
        if self.Cm_alpha is None:
            # Static-margin identity, Full_6DOF Sec 9.2:
            #   Cm_alpha = -CL_alpha * (xnp - xcg) / c_bar
            self.Cm_alpha = -self.CL_alpha * (self.xnp - self.xcg) / self.c_bar

        self.rotor_disk_area = np.pi * (self.rotor_diameter / 2.0) ** 2
        R = self.rotor_diameter / 2.0
        self.blade_solidity = self.blade_Nb * self.blade_chord / (np.pi * R)   # sigma, Sec 7.1

        # Rotor layout (H-frame, confirmed CAD geometry -- 6DOF.pdf Sec 2.1's
        # rotor-index/spin convention, with the ACTUAL per-crossbar spans):
        #   1 front-right (CCW, +y_front), 2 aft-left  (CCW, -y_aft),
        #   3 front-left  (CW,  -y_front), 4 aft-right (CW,  +y_aft)
        # Front and aft crossbars are different widths -- do not collapse
        # this back into a single symmetric arm.
        yf, ya = self.rotor_y_front, self.rotor_y_aft
        self.rotor_y_arm = np.array([yf, -ya, -yf, ya])

    @classmethod
    def from_json(cls, path):
        """Load overrides from a JSON file; unspecified fields keep the
        documented defaults above."""
        with open(path, "r") as f:
            overrides = json.load(f)
        # numpy arrays can't come straight from JSON lists for dataclass
        # fields with default_factory -- coerce known array fields
        for key in ("rotor_x_arm", "rotor_spin_sign"):
            if key in overrides:
                overrides[key] = np.asarray(overrides[key], dtype=float)
        return cls(**overrides)

    def to_json(self, path):
        # Only serialise genuine dataclass fields -- rotor_disk_area and
        # rotor_y_arm are DERIVED in __post_init__, not real parameters,
        # and must not be round-tripped through from_json (which passes
        # every key straight to the constructor).
        from dataclasses import fields
        field_names = {f.name for f in fields(self)}
        d = {}
        for k in field_names:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        with open(path, "w") as f:
            json.dump(d, f, indent=2)