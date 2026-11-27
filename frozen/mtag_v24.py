
"""v2.4 endcap likelihood material tag (preregistered).  Extension of the v2.2 endcap tag
to include inter-disk service material (cables, cooling, support structures) between the
9 existing sensor disks, adding 8 service-material bands for a total of 17 endcap bands.

The tag formula is identical to v2.2:
  p_conv(z) = sum_d c_d * [Phi((z - Z_d + HW_d)/s) - Phi((z - Z_d - HW_d)/s)] / (2 HW_d),
              c_d proportional to X0_d, sum c_d = 1
  p_sig(z)  = 1 / (Z_HI - Z_LO) mm^{-1}  (flat, signal-agnostic)
  lambda    = dchi2 / 2 + ln[p_sig(z) / p_conv(z)]

Only the DISKS list is extended: the 9 original sensor disks from v2.2 are preserved
unchanged, and 8 service bands are appended.  Service bands are placed at the midpoint
of each inter-disk gap, motivated by cable trays, cooling pipes, and support rings in
the CMS Phase-2 Inner Tracker endcap.

See the preregistration README for full motivation and geometry tables."""
import numpy as np
from scipy.special import ndtr

# ── Original 9 sensor disks (identical to v2.2 mtag_v22.py) ──
_SENSOR_DISKS = [
    (230,  0.006, 1.5),   # transition band: services, cables at barrel-endcap junction
    (300,  0.020, 3.0),   # inner pixel disk (TFPX-motivated)
    (450,  0.020, 3.0),   # inner pixel disk
    (620,  0.020, 3.0),   # inner pixel disk
    (820,  0.022, 3.5),   # inner pixel disk (edge)
    (1100, 0.030, 4.5),   # outer disk (TEPX/TEDD-motivated)
    (1500, 0.030, 5.0),   # outer disk
    (1900, 0.035, 5.5),   # outer disk
    (2300, 0.035, 6.0),   # outer disk
]

# ── 8 inter-disk service material bands (new in v2.4) ──
# Placed at the midpoint of each gap between adjacent sensor disks.
# Inner service bands (gaps 1-4, between TFPX-like disks): X0 = 0.005, HW = 1.5 mm
# Outer service bands (gaps 5-8, between TEPX/TEDD-like disks): X0 = 0.008, HW = 1.5 mm
_SERVICE_BANDS = [
    (265.0,  0.005, 1.5),  # midpoint of 230-300 gap (inner)
    (375.0,  0.005, 1.5),  # midpoint of 300-450 gap (inner)
    (535.0,  0.005, 1.5),  # midpoint of 450-620 gap (inner)
    (720.0,  0.005, 1.5),  # midpoint of 620-820 gap (inner)
    (960.0,  0.008, 1.5),  # midpoint of 820-1100 gap (outer)
    (1300.0, 0.008, 1.5),  # midpoint of 1100-1500 gap (outer)
    (1700.0, 0.008, 1.5),  # midpoint of 1500-1900 gap (outer)
    (2100.0, 0.008, 1.5),  # midpoint of 1900-2300 gap (outer)
]

# Combined 17-band geometry, sorted by z position
DISKS = sorted(_SENSOR_DISKS + _SERVICE_BANDS, key=lambda d: d[0])

Z_LO, Z_HI = 200.0, 2500.0       # fiducial z range for flat signal prior [mm]
P_FLOOR = 1e-300

# Oblique incidence reference (same as v2.2)
ETA_REF = 2.0
COS_THETA_REF = float(np.tanh(ETA_REF))     # ~ 0.964


def p_conv(z, s, disks=DISKS):
    """Normalised conversion probability density at vertex z, given resolution s = sigma_z."""
    Z = np.array([d[0] for d in disks], float)
    X = np.array([d[1] for d in disks], float)
    H = np.array([d[2] for d in disks], float)
    c = X / X.sum()
    z = np.asarray(z, float)[..., None]
    s = np.maximum(np.asarray(s, float), 1e-6)[..., None]
    return (c * (ndtr((z - Z + H) / s) - ndtr((z - Z - H) / s)) / (2 * H)).sum(-1)


def p_sig(z):
    """Flat signal prior over fiducial z range."""
    return np.full(np.shape(z), 1.0 / (Z_HI - Z_LO))


def lam(z, s, dchi2):
    """Endcap likelihood material tag lambda(z_m, sigma_z, Delta chi^2_3D)."""
    d = np.where(np.isfinite(dchi2), dchi2, 0.0)
    return 0.5 * d + np.log(p_sig(z)) - np.log(np.maximum(p_conv(z, s), P_FLOOR))


TGRID = (None, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 25.0)
