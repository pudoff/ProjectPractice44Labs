"""2D homogeneous transforms between coordinate spaces.

source_px - pixels of the original page (PNG/JPEG as decoded, PDF page rendered at the stated dpi)
work_px   - pixels of the prepared image (rotated, deskewed, resampled)
sheet_mm  - sheet millimetres, origin at the bottom-left corner of the sheet trim line, y up
"""
import math

import numpy as np


class Transform:
    def __init__(self, matrix=None):
        self.m = np.eye(3) if matrix is None else np.asarray(matrix, float).reshape(3, 3)

    @staticmethod
    def from_affine(a2x3):
        m = np.eye(3)
        m[:2] = np.asarray(a2x3, float)
        return Transform(m)

    @staticmethod
    def scale(sx, sy=None):
        return Transform(np.diag([sx, sx if sy is None else sy, 1.0]))

    @staticmethod
    def translate(tx, ty):
        m = np.eye(3)
        m[:2, 2] = tx, ty
        return Transform(m)

    def then(self, other):
        """Apply self first, then other."""
        return Transform(other.m @ self.m)

    def inverse(self):
        return Transform(np.linalg.inv(self.m))

    def apply(self, pts):
        p = np.asarray(pts, float)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        q = np.column_stack([p, np.ones(len(p))]) @ self.m.T
        q = q[:, :2] / q[:, 2:3]
        return q[0] if single else q

    def linear_scale(self):
        """Mean isotropic scale factor (length multiplier)."""
        return math.sqrt(abs(np.linalg.det(self.m[:2, :2])))

    def rotation_deg(self):
        return math.degrees(math.atan2(self.m[1, 0], self.m[0, 0]))

    def to_list(self):
        return [[round(float(v), 10) for v in row] for row in self.m]


def apply_to_geometry(t, geom):
    """Transform a primitive geometry dict (start/end/center/radius/vertices/angles) between spaces with a
    similarity transform. Angles are re-measured from transformed points so y-flips stay correct."""
    out = dict(geom)
    for key in ("start", "end", "center"):
        if geom.get(key) is not None:
            out[key] = t.apply(geom[key]).tolist()
    if geom.get("vertices") is not None:
        out["vertices"] = t.apply(geom["vertices"]).tolist()
    if geom.get("radius") is not None:
        out["radius"] = float(geom["radius"]) * t.linear_scale()
    if geom.get("start_angle") is not None and geom.get("center") is not None:
        c, r = np.asarray(geom["center"], float), float(geom["radius"])
        a0, a1 = math.radians(geom["start_angle"]), math.radians(geom["end_angle"])
        p0 = t.apply(c + r * np.array([math.cos(a0), math.sin(a0)]))
        p1 = t.apply(c + r * np.array([math.cos(a1), math.sin(a1)]))
        cc = np.asarray(out["center"])
        b0 = math.degrees(math.atan2(p0[1] - cc[1], p0[0] - cc[0])) % 360
        b1 = math.degrees(math.atan2(p1[1] - cc[1], p1[0] - cc[0])) % 360
        if np.linalg.det(t.m[:2, :2]) < 0:      # mirrored space: counter-clockwise arc runs the other way
            b0, b1 = b1, b0
        out["start_angle"], out["end_angle"] = b0, b1
    return out
