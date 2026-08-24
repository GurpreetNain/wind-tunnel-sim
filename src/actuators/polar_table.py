"""
Loader for XFLR5 polar export files (VLM/panel-method alpha sweeps).

Expected format (as exported by XFLR5's "Export Polar" for a Type-1
fixed-speed polar): a few header lines, then a column-header row
starting with "alpha", then whitespace-delimited numeric rows. Handles
Windows line endings (\\r\\n) as XFLR5 produces them.

This module only PARSES the file and caches it -- the decision of how
to use CL/CD/Cm at a given alpha (trust the table vs. fall back to the
flat-plate/blend model) lives in lifting_body.py, not here.
"""
import numpy as np
from functools import lru_cache


class PolarTable:
    """Holds one alpha-sweep polar, sorted ascending by alpha (radians)."""

    def __init__(self, alpha_rad, CL, CD, Cm):
        order = np.argsort(alpha_rad)
        self.alpha_rad = alpha_rad[order]
        self.CL = CL[order]
        self.CD = CD[order]
        self.Cm = Cm[order]
        self.alpha_min = float(self.alpha_rad[0])
        self.alpha_max = float(self.alpha_rad[-1])

    def interp(self, alpha_rad):
        """Linear interpolation; CLAMPS to the edge value outside the
        table's range (np.interp's default behavior) -- callers that
        want a blended post-stall extension must check
        alpha_min/alpha_max themselves and blend, not rely on this
        clamp being physically meaningful on its own."""
        CL = np.interp(alpha_rad, self.alpha_rad, self.CL)
        CD = np.interp(alpha_rad, self.alpha_rad, self.CD)
        Cm = np.interp(alpha_rad, self.alpha_rad, self.Cm)
        return CL, CD, Cm

    def in_range(self, alpha_rad):
        return self.alpha_min <= alpha_rad <= self.alpha_max


def _parse_xflr5_text(path):
    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("alpha") and "CL" in stripped:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"Could not find the 'alpha ... CL ...' column-header row in "
            f"{path}. Is this a raw XFLR5 polar export?")

    col_names = lines[header_idx].split()
    data_lines = [l for l in lines[header_idx + 1:] if l.strip()]
    if not data_lines:
        raise ValueError(f"No data rows found after the header in {path}.")

    data = np.array([[float(x) for x in l.split()] for l in data_lines])
    cols = {name: data[:, i] for i, name in enumerate(col_names)}

    for required in ("alpha", "CL", "CD", "Cm"):
        if required not in cols:
            raise ValueError(
                f"Column '{required}' not found in {path}. "
                f"Columns present: {list(cols.keys())}")
    return cols


@lru_cache(maxsize=8)
def load_xflr5_polar(path):
    """
    Parses an XFLR5 polar export and returns a PolarTable. Cached by
    path (lru_cache) so repeated calls during a simulation run don't
    re-read/re-parse the file every timestep -- call this once, e.g. in
    InterceptorParams, not inside the per-step aero_coefficients() call.
    """
    cols = _parse_xflr5_text(path)
    alpha_rad = np.radians(cols["alpha"])
    return PolarTable(alpha_rad, cols["CL"], cols["CD"], cols["Cm"])