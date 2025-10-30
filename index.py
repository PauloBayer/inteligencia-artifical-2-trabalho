# index.py
# ------------------------------------------------------------
# Single-ball controller — NN decides radius-change magnitude
# Advanced init + robust anti-stall scanning + ring-based radius sign.
#
# New in this version:
#  - Bigger traced-step budget + IoU-chasing extension to push toward IoU≈1.0 (eval only)
#  - Optional IoU-based early stop when near-1.0 on annotated data
#  - Border-aware expansion: when touching image edges, we re-center minimally and THEN expand
#  - Overshoot taming:
#       * Approach-step uses <1 gain and clamps max px
#       * Momentum is damped when entering approach/border_seek
#       * Radius deltas capped tighter when near the centroid
# ------------------------------------------------------------

import os
import json
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from gapy import gago, bits2bytes

# ============================================================
# 0) Config
# ============================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
rng = np.random.default_rng(GLOBAL_SEED)

IN_SIZE  = 28
IMG_SIZE = 255  # 255x255
R_NORM   = int(math.ceil(math.hypot(IMG_SIZE - 1, IMG_SIZE - 1)))  # ~360

DATA_ROOT = "amostras/dados"
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, "annotations.jsonl")

# --- Pixel thresholding for "black" on the full-res grayscale image ---
BLACK_THR = 64          # treat <=64 as black (tune: 32..96)

# --- Perfect-fit stop (exact ring match) ---
STOP_ON_PERFECT      = True
PERFECT_INNER_FRAC   = 0.995    # was 1.00
PERFECT_OUTER_FRAC   = 0.005    # was 0.00
PERFECT_THICKNESS    = 1        # average ±1 px to reduce aliasing brittleness

# --- GIF overlay flags ---
SHOW_GT_IN_GIF       = True     # False removes the green GT circle from GIFs
PRED_ON_TOP          = True     # draw red prediction after green GT so red overlays

# --- GIF overlay flags ---
SHOW_GT_IN_GIF       = True      # False removes the green GT circle from GIFs
PRED_ON_TOP          = True      # draw red prediction after green GT so red overlays

# --- Trace/GIF trimming (ensure GIF stops exactly where controller stopped) ---
TRIM_GIF_AT_STOP     = True      # cut frames after first stop (IoU/perfect)
KEEP_SNAP_AFTER_STOP = True      # if a 'snap_refine' frame exists at same t, keep it

# --- GIF HUD (text) settings ---
HUD_TEXT_COLOR   = (255, 255, 255)   # text color (e.g., white)
HUD_STROKE_COLOR = (0, 0, 0)         # outline color (e.g., black)
HUD_BG_RGBA      = None              # e.g., (0, 0, 0, 120) for translucent box, or None for no box
HUD_FONT_PATH    = None              # path to .ttf/.otf; None => default PIL font
HUD_FONT_SIZE    = 10                # base pt size; will be scaled by GIF_SCALE
HUD_PAD          = 2                 # padding (in px, pre-scale) around the text box
HUD_POS          = (5, 5)            # top-left position (pre-scale)

# --- Anti-stall on white frames: step growth + jitter per streak length ---
WHITE_STREAK_GROWTH   = 0.80
WHITE_JITTER_PX       = 3
WHITE_JITTER_CLAMP    = 24
WHITE_SUPERJUMP_EVERY = 10
WHITE_SUPERJUMP_PX    = IMG_SIZE // 2

# --- White-scan radius growth (let NN expand after long white streak) ---
WHITE_RADIUS_GROW_AFTER    = 20   # allow growth after this many consecutive all-white steps
WHITE_RADIUS_GROW_EVERY    = 4    # once allowed, grow again every N additional white steps
WHITE_RADIUS_GROW_MAX_PX   = 6    # cap per growth event (pixels)
WHITE_RADIUS_REQUIRE_BIT   = True # require sr==+1 (NN "wants" to expand) for growth
WHITE_RADIUS_RESET_STREAK  = False# optionally reset white_streak after growth

# --- Approach overshoot tamers ---
APPROACH_STEP_GAIN_FAR     = 0.60   # fraction of distance when far
APPROACH_STEP_GAIN_NEAR    = 0.35   # fraction when near
APPROACH_NEAR_FRAC_OF_R    = 0.75   # "near" if dist < 0.75 * r
APPROACH_STEP_MAX_PX       = 32     # clamp approach step
BORDER_STEP_MAX_PX         = 24     # clamp border_seek step
MOMENTUM_DAMP_ON_MODE_CHANGE = 0.25 # multiply vx,vy when entering approach/border_seek

# --- Minimal heuristics for radius sign from rings (not magnitude) ---
EPS_OUTER_EXPAND = 0.02  # if >2% black on r+1 ring → expand (we're cutting the edge)
EPS_INNER_SHRINK = 0.40  # if inner ring <40% black (and interior has black) → shrink

# --- Distance-aware radius caps (limit NN's |Δr| near the target) ---
RAD_CAP_NEAR_PX = 3
RAD_CAP_FAR_PX  = 8

# --- IoU chase / eval (only when GT is known, e.g., traced run) ---
IOU_EVAL_STOP      = True
IOU_EVAL_THRESH    = 1.0
IOU_CHASE_ENABLE   = True
IOU_IMPROVE_DELTA  = 0.001
IOU_STEPS_BONUS    = 64
IOU_EXTRA_CAP      = 2000

# ============================================================
# 1) Hyperparameters / presets
# ============================================================
CTRL_STEPS_COARSE = 8
CTRL_STEPS_FINE   = 12
PATIENCE_STEPS    = 4
WARMUP_STEPS      = 3
IMPROVE_EPS       = 1e-6

RING_SAMPLES_COARSE = 128
RING_SAMPLES_FINE   = 256

PROBE_R_COARSE   = 48
PROBE_R_FINE     = 64
PROBE_THICKNESS  = 2
W_PROBE_COARSE   = 0.35
W_PROBE_FINE     = 0.10

TH_INNER = 0.90
TH_OUTER = 0.10

RAD_STEP_FLOOR_PX   = 1
TH_EXPAND_INNER     = 0.85  # (legacy)
TH_EXPAND_OUTER     = 0.50  # (legacy)

AUTO_EXTEND_STEPS       = True
EXTRA_STEPS_HAS_SIGNAL  = 8
EXTRA_STEPS_BORDER      = 6
EXTRA_STEPS_REFINE      = 4
EXTRA_STEPS_CAP         = 24
EXTRA_STEPS_CAP_TRACE   = 2000

GA_POP_COARSE   = 120
GA_GENS_COARSE  = 22
GA_MUT_COARSE   = 0.90
GA_POP_POLISH   = GA_POP_COARSE
GA_GENS_POLISH  = 6
GA_MUT_POLISH   = 0.90
ELITE_COUNT     = 2

import os as _os
EARLY_STOP = False
SEARCH_BUDGET = _os.getenv("NN_BUDGET", "thorough")
PRESETS = {
    "fast":      {"CTRL_STEPS_COARSE": 6,  "CTRL_STEPS_FINE": 10,  "PATIENCE_STEPS": 2,  "EXTRA_STEPS_CAP": 12},
    "balanced":  {"CTRL_STEPS_COARSE": 8,  "CTRL_STEPS_FINE": 12,  "PATIENCE_STEPS": 4,  "EXTRA_STEPS_CAP": 24},
    "thorough":  {"CTRL_STEPS_COARSE": 24, "CTRL_STEPS_FINE": 64,  "PATIENCE_STEPS": 8,  "EXTRA_STEPS_CAP": 256},
    "max":       {"CTRL_STEPS_COARSE": 48, "CTRL_STEPS_FINE": 128, "PATIENCE_STEPS": 12, "EXTRA_STEPS_CAP": 2000},
}
_p = PRESETS.get(SEARCH_BUDGET, PRESETS["max"])
CTRL_STEPS_COARSE = _p["CTRL_STEPS_COARSE"]
CTRL_STEPS_FINE   = _p["CTRL_STEPS_FINE"]
PATIENCE_STEPS    = _p["PATIENCE_STEPS"]
EXTRA_STEPS_CAP   = _p["EXTRA_STEPS_CAP"]

# make traced pass wander longer by default
INFER_STEPS_MULT = int(_os.getenv("NN_INFER_MULT", "3"))

# ============================================================
# 1.2) Init / decode / movement
# ============================================================
INIT_POLICY            = "random_then_nn"
INIT_RANDOM_R_MIN_PX   = 4
INIT_RANDOM_R_MAX_FRAC = 0.45
INIT_NN_STEPS          = 1
INIT_STEP_ABS_SCALE    = IMG_SIZE
USE_INIT_HEAD          = False

USE_GRAY_CODE     = True
MOVE_GAMMA        = 0.85
RAD_GAMMA         = 0.85

MOMENTUM_BETA     = 0.6
DEADZONE_PX       = 0

WHITE_USE_NN_STEP      = True
WHITE_STEP_ABS_MAX     = IMG_SIZE // 2
WHITE_STUCK_PATIENCE   = 2
WHITE_JUMP_PX          = IMG_SIZE // 3

RUNS_DIR          = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_ID            = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_JSONL_PATH    = os.path.join(RUNS_DIR, f"run_{RUN_ID}.jsonl")

MAKE_GIFS         = True
GIFS_DIR          = os.path.join(RUNS_DIR, "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)
GIF_SCALE         = 2
# --- GIF timing ---
GIF_DURATION_MS        = 120       # already in your code
GIF_TAIL_HOLD_MS       = 5000      # keep the LAST frame visible for 5 seconds

# Some viewers ignore a very long delay on the final frame. Enable this to duplicate
# the last frame into N short frames that add up to GIF_TAIL_HOLD_MS.
GIF_TAIL_COMPAT_DUPLICATE    = False
GIF_TAIL_DUPLICATE_EACH_MS   = 250   # used only if GIF_TAIL_COMPAT_DUPLICATE=True

GIF_LIMIT         = None

# ============================================================
# 2) I/O
# ============================================================
def load_image_small_bin(file_path, out_size=IN_SIZE):
    img = Image.open(file_path).convert('L').resize((out_size, out_size), Image.NEAREST)
    arr = np.array(img)
    return np.where(arr < 128, 1.0, 0.0).astype(np.float32)

def load_image_full_gray(file_path):
    return np.array(Image.open(file_path).convert('L'))

# ============================================================
# 3) Sampling / metrics
# ============================================================
def _precompute_trig(n_samples):
    theta = (2.0 * np.pi) * (np.arange(n_samples, dtype=np.float32) / float(n_samples))
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

COS_COARSE, SIN_COARSE = _precompute_trig(RING_SAMPLES_COARSE)
COS_FINE,   SIN_FINE   = _precompute_trig(RING_SAMPLES_FINE)

def _ring_coords(cx, cy, rr, cos_tab, sin_tab, size):
    rr = int(max(1, abs(int(round(rr)))))
    xs = np.rint(cx + rr * cos_tab).astype(np.int32)
    ys = np.rint(cy + rr * sin_tab).astype(np.int32)
    valid = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    return xs[valid], ys[valid]

def _ring_fraction_vec(img255, cx, cy, r, delta, cos_tab, sin_tab):
    size = img255.shape[0]
    rr = int(round(abs((r + delta) if r != 0 else delta)))
    xs, ys = _ring_coords(cx, cy, rr, cos_tab, sin_tab, size)
    if xs.size == 0:
        return 0.0
    vals = img255[ys, xs]
    black = np.count_nonzero(vals <= BLACK_THR)
    total = int(xs.size)
    return black / float(total)

def _ring_fraction_thick(img255, cx, cy, r, delta_center, thickness, cos_tab, sin_tab):
    if thickness <= 0:
        return _ring_fraction_vec(img255, cx, cy, r, delta_center, cos_tab, sin_tab)
    acc = 0.0; cnt = 0
    for u in range(-thickness, thickness + 1):
        acc += _ring_fraction_vec(img255, cx, cy, r, delta_center + u, cos_tab, sin_tab)
        cnt += 1
    return acc / float(cnt) if cnt > 0 else 0.0

def _border_cut_vec(img255, cx, cy, r, cos_tab, sin_tab):
    return _ring_fraction_vec(img255, cx, cy, r, delta=0, cos_tab=cos_tab, sin_tab=sin_tab)

def _circle_mask(size, cx, cy, r):
    yy, xx = np.ogrid[:size, :size]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

def interior_fill_fraction(img255, cx, cy, r):
    mask = _circle_mask(img255.shape[0], cx, cy, r)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    filled = int(np.count_nonzero(img255[mask] <= BLACK_THR))
    return filled / float(area)

def iou_circle(size, c1, c2):
    m1 = _circle_mask(size, c1[0], c1[1], c1[2])
    m2 = _circle_mask(size, c2[0], c2[1], c2[2])
    inter = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    return (inter / union) if union > 0 else 0.0

# ============================================================
# 3.b) Discrete-mask IoU and snap refinement (add this block)
# ============================================================
def black_mask(img255, thr=BLACK_THR):
    # Binary mask of the actual black region in the image.
    return (img255 <= thr)

def iou_circle_vs_mask(img255, cx, cy, r, thr=BLACK_THR):
    size = img255.shape[0]
    cm = _circle_mask(size, int(cx), int(cy), int(max(1, r)))
    bm = black_mask(img255, thr)
    inter = int(np.count_nonzero(cm & bm))
    union = int(np.count_nonzero(cm | bm))
    return (inter / union) if union > 0 else 0.0

def snap_refine_mask_iou(img255, cx, cy, r, *,
                         dxy=1, dr=2, thr=BLACK_THR,
                         prefer_smaller_radius=True):
    """
    Local brute-force search to snap (cx,cy,r) to the actual rasterized blob.
    Searches a tiny neighborhood and maximizes IoU to the image's black mask.
    Tie-break prefers smaller radii (eliminates r→r+1 bias on crisp disks).
    """
    size = img255.shape[0]
    best_score = iou_circle_vs_mask(img255, cx, cy, r, thr)
    best = (int(cx), int(cy), int(r))
    for dy in range(-dxy, dxy + 1):
        for dx in range(-dxy, dxy + 1):
            cx2 = int(np.clip(cx + dx, 0, size - 1))
            cy2 = int(np.clip(cy + dy, 0, size - 1))
            # keep radius within feasible bounds for this center
            rmax2 = r_fit_for_center(size, cx2, cy2)
            for dr_ in range(-dr, dr + 1):
                r2 = int(np.clip(r + dr_, 1, rmax2))
                s = iou_circle_vs_mask(img255, cx2, cy2, r2, thr)
                if (s > best_score) or (abs(s - best_score) < 1e-12 and prefer_smaller_radius and r2 < best[2]):
                    best_score = s
                    best = (cx2, cy2, r2)
    return best, best_score

# ============================================================
# 4) Probe + loss
# ============================================================
def _make_probe_list(base_r, img_size):
    lst = [int(base_r), int(2*base_r), int(3*base_r)]
    max_r = int(0.45 * img_size)
    return [r for r in lst if r >= 2 and r <= max_r] or [max(2, min(lst))]

def _probe_max_thick(img255, cx, cy, probe_r_list, thickness, cos_tab, sin_tab):
    if not probe_r_list:
        return 0.0
    vals = [
        _ring_fraction_thick(img255, cx, cy, r=0, delta_center=pr, thickness=thickness,
                             cos_tab=cos_tab, sin_tab=sin_tab)
        for pr in probe_r_list
    ]
    return max(vals) if vals else 0.0

def make_metrics_loss(img255, cos_tab, sin_tab, cache_dict, probe_r_list, weights=None, w_probe=0.20):
    if weights is None:
        w_fill, w_inner, w_outer, w_cut = 0.8, 1.0, 1.0, 0.10
    else:
        w_fill, w_inner, w_outer, w_cut = weights
    def metrics_loss(cx, cy, r):
        key = (int(cx), int(cy), int(r))
        hit = cache_dict.get(key)
        if hit is not None:
            return hit
        fill = interior_fill_fraction(img255, cx, cy, r)
        inner_black = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_black = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)
        cut  = _border_cut_vec(img255, cx, cy, r, cos_tab=cos_tab, sin_tab=sin_tab)
        probe_black = _probe_max_thick(img255, cx, cy, probe_r_list, PROBE_THICKNESS, cos_tab, sin_tab)
        loss = (
            w_fill  * (1.0 - fill)**2 +
            w_inner * (1.0 - inner_black)**2 +
            w_outer * (outer_black)**2 +
            w_cut   * (cut**2) +
            w_probe * (1.0 - probe_black)**2
        )
        loss = float(loss)
        cache_dict[key] = loss
        return loss
    return metrics_loss

# ============================================================
# 5) Tiny MLP
# ============================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, use_init_head=False):
        self.hidden_sizes = hidden_sizes
        self.use_init_head = use_init_head
        self.weights = []
        self.biases = []
        prev = input_size
        for h in hidden_sizes:
            self.weights.append((np.random.randn(prev, h).astype(np.float32) * 0.01))
            self.biases.append((np.random.randn(h).astype(np.float32) * 0.01))
            prev = h
        self.weights.append((np.random.randn(prev, output_size).astype(np.float32) * 0.01))
        self.biases.append((np.random.randn(output_size).astype(np.float32) * 0.01))

    @staticmethod
    def sigmoid(x):
        x = np.asarray(x, dtype=np.float32)
        z = np.empty_like(x, dtype=np.float32)
        pos = x >= 0; neg = ~pos
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg]); z[neg] = ex / (1.0 + ex)
        return z

    def forward(self, x):
        a = x
        for i in range(len(self.hidden_sizes)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
        out = self.sigmoid(np.dot(a, self.weights[-1]) + self.biases[-1])
        return out

    def get_weights(self):
        vec = []
        for w, b in zip(self.weights, self.biases):
            vec.append(w.flatten()); vec.append(b)
        return np.concatenate(vec).astype(np.float32)

    def set_weights(self, vector):
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            n_w = int(np.prod(w_shape))
            self.weights[i] = vector[idx:idx+n_w].reshape(w_shape).astype(np.float32); idx += n_w
            n_b = int(np.prod(b_shape))
            self.biases[i] = vector[idx:idx+n_b].astype(np.float32); idx += n_b

ACTION_BITS = 22
def split_outputs(out):
    if out.shape[0] <= ACTION_BITS:
        return out, None
    return out[:ACTION_BITS], out[ACTION_BITS:]

# ============================================================
# 6) 22-bit decode (Gray code)
# ============================================================
def _bits_to_uint8_lsb(bits8):
    v = 0
    for i, b in enumerate(bits8):
        v |= (int(b) << i)
    return v & 0xFF

def _gray_to_binary_u8(g):
    g = int(g) & 0xFF
    b = g
    shift = 1
    while shift < 8:
        b ^= (b >> shift); shift <<= 1
    return b & 0xFF

def _decode_u8(bits8, use_gray=True):
    raw = _bits_to_uint8_lsb(bits8)
    return _gray_to_binary_u8(raw) if use_gray else raw

def _smooth_frac(u8, gamma=1.0):
    x = max(0.0, min(255.0, float(u8))) / 255.0
    return x**gamma if gamma != 1.0 else x

def decode_actions(out_vec, r_curr):
    bits_all = (out_vec > 0.5).astype(np.uint8)
    bits = bits_all[:ACTION_BITS]
    bx_pos, bx_neg, by_pos, by_neg = map(int, bits[0:4])
    sx = 1 if (bx_pos and not bx_neg) else (-1 if (bx_neg and not bx_pos) else 0)
    sy = 1 if (by_pos and not by_neg) else (-1 if (by_neg and not by_pos) else 0)
    k_move_u8 = _decode_u8(bits[4:12], use_gray=USE_GRAY_CODE)
    move_frac = _smooth_frac(k_move_u8, gamma=MOVE_GAMMA)
    move_step = float(move_frac) * float(max(1, r_curr))
    br_up, br_down = int(bits[12]), int(bits[13])
    sr = 1 if (br_up and not br_down) else (-1 if (br_down and not br_up) else 0)
    k_rad_u8 = _decode_u8(bits[14:22], use_gray=USE_GRAY_CODE)
    rad_frac = _smooth_frac(k_rad_u8, gamma=RAD_GAMMA)
    rad_step = float(rad_frac) * float(max(1, r_curr))
    return sx, sy, move_step, sr, rad_step, bits

# ============================================================
# 7) State / init / bounds
# ============================================================
def build_input_vec(img_small_bin, cx, cy, r):
    state = np.array([cx/IMG_SIZE, cy/IMG_SIZE, r/float(R_NORM)], dtype=np.float32)
    return np.concatenate([img_small_bin.flatten(), state], axis=0).astype(np.float32)

def r_fit_for_center(size, cx, cy):
    return int(max(0, min(cx, cy, size-1-cx, size-1-cy)))

def initial_center_fit_all(size):
    cx = size // 2; cy = size // 2
    r  = r_fit_for_center(size, cx, cy)
    return cx, cy, r

def clamp_center_with_radius(size, cx, cy, r):
    cx = int(np.clip(cx, r, size-1-r))
    cy = int(np.clip(cy, r, size-1-r))
    return cx, cy

def enforce_bounds(size, cx, cy, r):
    rmax = r_fit_for_center(size, cx, cy)
    r = int(max(1, min(r, rmax)))
    cx, cy = clamp_center_with_radius(size, cx, cy, r)
    return cx, cy, r

def initial_center_random(size, rng, r_min_px=INIT_RANDOM_R_MIN_PX, r_max_frac=INIT_RANDOM_R_MAX_FRAC):
    r_max_px = int(max(1, min(int(r_max_frac * size), size // 2)))
    r = int(rng.integers(low=max(1, r_min_px), high=r_max_px + 1))
    cx = int(rng.integers(low=r, high=size - r))
    cy = int(rng.integers(low=r, high=size - r))
    return cx, cy, r

def nn_radius_delta(sr, rad_step):
    step = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step))))
    return int(sr) * step

def apply_radius(size, cx, cy, r, delta_r):
    r_new = max(1, r + int(delta_r))
    cx, cy, r_new = enforce_bounds(size, cx, cy, r_new)
    return cx, cy, r_new

def apply_radius_recenter(size, cx, cy, r, delta_r):
    desired = max(1, r + int(delta_r))
    cx = int(np.clip(cx, desired, size - 1 - desired))
    cy = int(np.clip(cy, desired, size - 1 - desired))
    r_new = desired
    cx, cy, r_new = enforce_bounds(size, cx, cy, r_new)
    return cx, cy, r_new

def nn_initial_adjust(nn, img_small_bin, cx, cy, r, size=IMG_SIZE):
    x_in = build_input_vec(img_small_bin, cx, cy, r)
    out  = nn.forward(x_in)
    act, _ = split_outputs(out)
    sx, sy, move_step_nn, sr, rad_step_nn, _bits = decode_actions(act, max(1, r))
    move_frac = float(move_step_nn) / float(max(1, r))
    step_abs  = int(round(move_frac * float(max(1, INIT_STEP_ABS_SCALE - 1))))
    dx = int(sx) * step_abs
    dy = int(sy) * step_abs
    cx = int(cx + dx); cy = int(cy + dy)
    cx, cy = clamp_center_with_radius(size, cx, cy, r)
    delta_r = nn_radius_delta(sr, rad_step_nn)
    cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
    cx, cy, r = enforce_bounds(size, cx, cy, r)
    return cx, cy, r

def nn_init_head_propose(nn, img_small_bin, size, r_min_px=INIT_RANDOM_R_MIN_PX):
    cx_seed, cy_seed, r_seed = initial_center_fit_all(size)
    x_in = build_input_vec(img_small_bin, cx_seed, cy_seed, r_seed)
    out  = nn.forward(x_in)
    act, init = split_outputs(out)
    if init is None or init.shape[0] < 3:
        return initial_center_fit_all(size)
    u_cx = float(init[0]); u_cy = float(init[1]); u_r = float(init[2])
    cx = int(round(u_cx * (size - 1)))
    cy = int(round(u_cy * (size - 1)))
    rmax = r_fit_for_center(size, cx, cy)
    rmin = int(max(1, r_min_px))
    if rmax < rmin:
        cx, cy, _ = initial_center_fit_all(size)
        rmax = r_fit_for_center(size, cx, cy)
    r = int(round(rmin + u_r * max(0, (rmax - rmin))))
    cx, cy, r = enforce_bounds(size, cx, cy, r)
    return cx, cy, r

def choose_initial_state(nn, img255, img_small_bin, size, rng, policy=INIT_POLICY, use_init_head=USE_INIT_HEAD):
    if policy == "center_fit":
        cx, cy, r = initial_center_fit_all(size)
    elif policy == "random_only":
        cx, cy, r = initial_center_random(size, rng)
    elif policy == "nn_only":
        cx, cy, r = initial_center_fit_all(size)
        for _ in range(int(INIT_NN_STEPS)):
            cx, cy, r = nn_initial_adjust(nn, img_small_bin, cx, cy, r, size)
    elif policy == "random_then_nn":
        cx, cy, r = initial_center_random(size, rng)
        for _ in range(int(INIT_NN_STEPS)):
            cx, cy, r = nn_initial_adjust(nn, img_small_bin, cx, cy, r, size)
    elif policy == "nn_head_only" and use_init_head:
        cx, cy, r = nn_init_head_propose(nn, img_small_bin, size)
    elif policy == "random_then_nn_head" and use_init_head:
        _cx, _cy, _r = initial_center_random(size, rng)
        cx, cy, r = nn_init_head_propose(nn, img_small_bin, size)
    else:
        cx, cy, r = initial_center_fit_all(size)
    return enforce_bounds(size, cx, cy, r)

# ============================================================
# 8) Heuristic helpers
# ============================================================
def any_black_interior(img255, cx, cy, r):
    mask = _circle_mask(img255.shape[0], cx, cy, r)
    if not np.any(mask): return False
    return np.any(img255[mask] <= BLACK_THR)

def border_black_direction(img255, cx, cy, r, cos_tab, sin_tab):
    size = img255.shape[0]
    rr = int(max(1, abs(int(round(r)))))
    xs, ys = _ring_coords(cx, cy, rr, cos_tab, sin_tab, size)
    if xs.size == 0: return None
    mask_black = (img255[ys, xs] <= BLACK_THR)
    if not np.any(mask_black): return None
    xb = xs[mask_black].astype(np.float32)
    yb = ys[mask_black].astype(np.float32)
    vx = float(np.mean(xb - cx)); vy = float(np.mean(yb - cy))
    norm = math.hypot(vx, vy)
    if norm == 0.0: return None
    return (vx / norm, vy / norm)

def circle_complete(img255, cx, cy, r, cos_tab, sin_tab, th_inner=TH_INNER, th_outer=TH_OUTER):
    inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
    outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)
    return (inner_b >= th_inner) and (outer_b <= th_outer), inner_b, outer_b

def centroid_black_interior(img255, cx, cy, r):
    size = img255.shape[0]
    mask = _circle_mask(size, cx, cy, r)
    ys, xs = np.where(mask & (img255 <= BLACK_THR))
    if xs.size == 0: return None
    mx = int(np.clip(int(np.rint(xs.mean())), 0, size - 1))
    my = int(np.clip(int(np.rint(ys.mean())), 0, size - 1))
    return (mx, my)

def circle_perfect(img255, cx, cy, r, cos_tab, sin_tab,
                   inner_req=PERFECT_INNER_FRAC, outer_req=PERFECT_OUTER_FRAC):
    if PERFECT_THICKNESS > 0:
        inner_b = _ring_fraction_thick(img255, cx, cy, r, delta_center=-0.5, thickness=PERFECT_THICKNESS,
                                       cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_thick(img255, cx, cy, r, delta_center=+0.5, thickness=PERFECT_THICKNESS,
                                       cos_tab=cos_tab, sin_tab=sin_tab)
    else:
        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)
    is_perfect = (inner_b >= inner_req) and (outer_b <= outer_req)
    return is_perfect, inner_b, outer_b

# ============================================================
# 9) Controller (no tracing)
# ============================================================
def run_controller(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, probe_r_list,
                   return_initial=False):
    size = img255.shape[0]
    cx, cy, r = choose_initial_state(nn, img255, img_small_bin, size, rng,
                                     policy=INIT_POLICY, use_init_head=USE_INIT_HEAD)

    initial_loss = metrics_loss_fn(cx, cy, r)
    initial_state = (cx, cy, r, initial_loss)

    best = (initial_loss, cx, cy, r)
    no_improve = 0

    vx = 0.0; vy = 0.0
    scan_dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    scan_k = 0
    stuck_white = 0
    white_streak = 0
    prev_mode = "init"
    last_dist = None

    t = 0
    max_steps = steps
    max_steps_cap = steps + EXTRA_STEPS_CAP

    while t < max_steps:
        x_in = build_input_vec(img_small_bin, cx, cy, r)
        out  = nn.forward(x_in)
        act, _ = split_outputs(out)
        sx, sy, move_step_nn, sr, rad_step_nn, _bits = decode_actions(act, r)

        interior_black = any_black_interior(img255, cx, cy, r)
        border_dir = border_black_direction(img255, cx, cy, r, cos_tab, sin_tab)
        border_has_black = (border_dir is not None)
        all_white = (not interior_black) and (not border_has_black)

        # --- movement selection ---
        if all_white:
            if sx == 0 and sy == 0:
                sx, sy = scan_dirs[scan_k % 4]; scan_k += 1
            if (sx > 0 and cx >= size-1-r) or (sx < 0 and cx <= r):   sx = -sx
            if (sy > 0 and cy >= size-1-r) or (sy < 0 and cy <= r):   sy = -sy
            if WHITE_USE_NN_STEP:
                move_frac = float(move_step_nn) / float(max(1, r))
                base_step = int(round(max(1.0, min(WHITE_STEP_ABS_MAX, move_frac * WHITE_STEP_ABS_MAX))))
            else:
                base_step = max(1, int(r))
            white_streak += 1
            grown = int(round(base_step * (1.0 + WHITE_STREAK_GROWTH * white_streak)))
            step  = int(min(WHITE_STEP_ABS_MAX, grown))
            jitter_cap = int(min(WHITE_JITTER_CLAMP, WHITE_JITTER_PX * white_streak))
            jx = int(rng.integers(-jitter_cap, jitter_cap + 1)) if jitter_cap > 0 else 0
            jy = int(rng.integers(-jitter_cap, jitter_cap + 1)) if jitter_cap > 0 else 0
            raw_dx = step * sx + jx
            raw_dy = step * sy + jy
            mode   = "white_scan"
            if white_streak % max(1, WHITE_SUPERJUMP_EVERY) == 0:
                tx = int(rng.integers(r, size - r)); ty = int(rng.integers(r, size - r))
                dx_f = float(tx - cx); dy_f = float(ty - cy); dist = math.hypot(dx_f, dy_f)
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    raw_dx = WHITE_SUPERJUMP_PX * ux + (rng.integers(-WHITE_JITTER_PX, WHITE_JITTER_PX + 1))
                    raw_dy = WHITE_SUPERJUMP_PX * uy + (rng.integers(-WHITE_JITTER_PX, WHITE_JITTER_PX + 1))
                    mode = "white_superjump"
            last_dist = None

        elif (not interior_black) and border_has_black:
            white_streak = 0
            step = int(max(1, min(BORDER_STEP_MAX_PX, int(r))))
            ux, uy = border_dir
            raw_dx = step * ux; raw_dy = step * uy
            mode = "border_seek"
            last_dist = None

        else:
            white_streak = 0
            cen = centroid_black_interior(img255, cx, cy, r)
            if cen is not None:
                tx, ty = cen
                dx_f = float(tx - cx); dy_f = float(ty - cy)
                dist = math.hypot(dx_f, dy_f)
                last_dist = dist
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    near = (dist < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
                    gain = (APPROACH_STEP_GAIN_NEAR if near else APPROACH_STEP_GAIN_FAR)
                    step = int(max(1, min(APPROACH_STEP_MAX_PX, round(dist * gain))))
                    raw_dx = step * ux; raw_dy = step * uy
                else:
                    raw_dx = 0.0; raw_dy = 0.0
            else:
                raw_dx = 0.0; raw_dy = 0.0
                last_dist = None
            mode = "approach"

        # momentum damping on mode change into approach/border_seek
        if mode in ("approach", "border_seek") and mode != prev_mode:
            vx *= MOMENTUM_DAMP_ON_MODE_CHANGE
            vy *= MOMENTUM_DAMP_ON_MODE_CHANGE

        # momentum + clamp
        if abs(raw_dx) <= DEADZONE_PX: raw_dx = 0.0
        if abs(raw_dy) <= DEADZONE_PX: raw_dy = 0.0
        vx = MOMENTUM_BETA * vx + (1.0 - MOMENTUM_BETA) * raw_dx
        vy = MOMENTUM_BETA * vy + (1.0 - MOMENTUM_BETA) * raw_dy
        dx = int(round(vx)); dy = int(round(vy))
        new_cx = cx + dx; new_cy = cy + dy
        new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)

        if mode.startswith("white_") and (new_cx == cx and new_cy == cy):
            stuck_white += 1
            if stuck_white >= WHITE_STUCK_PATIENCE:
                gx, gy = size // 2, size // 2
                dx_f = float(gx - cx); dy_f = float(gy - cy)
                dist = math.hypot(dx_f, dy_f)
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    vx = (1.0 - MOMENTUM_BETA) * WHITE_JUMP_PX * ux
                    vy = (1.0 - MOMENTUM_BETA) * WHITE_JUMP_PX * uy
                    dx = int(round(vx)); dy = int(round(vy))
                    new_cx = cx + dx; new_cy = cy + dy
                    new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        # ---------------- WHITE-SCAN RADIUS GROWTH (NEW) ----------------
        white_growth_done = False
        if all_white:
            grow_ready = (white_streak >= WHITE_RADIUS_GROW_AFTER)
            if grow_ready:
                periodic_ok = ((white_streak - WHITE_RADIUS_GROW_AFTER) % max(1, WHITE_RADIUS_GROW_EVERY) == 0)
                bit_ok = (sr > 0) if WHITE_RADIUS_REQUIRE_BIT else True
                if periodic_ok and bit_ok:
                    raw = abs(nn_radius_delta(sr, rad_step_nn))
                    grow_px = int(min(max(RAD_STEP_FLOOR_PX, raw), WHITE_RADIUS_GROW_MAX_PX))
                    if grow_px > 0:
                        cx, cy, r = apply_radius_recenter(size, cx, cy, r, +grow_px)
                        cx, cy, r = enforce_bounds(size, cx, cy, r)
                        if WHITE_RADIUS_RESET_STREAK:
                            white_streak = 0
                        white_growth_done = True
        # ----------------------------------------------------------------

        # --- radius control with distance-aware caps (skip if we already grew) ---
        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)

        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if outer_b > EPS_OUTER_EXPAND:
                raw = abs(nn_radius_delta(sr, rad_step_nn))
                delta_r = +min(raw, cap_px)
                cx, cy, r = apply_radius_recenter(size, cx, cy, r, delta_r)
            elif interior_black and (inner_b < EPS_INNER_SHRINK):
                raw = abs(nn_radius_delta(sr, rad_step_nn))
                delta_r = -min(raw, cap_px)
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)

        cx, cy, r = enforce_bounds(size, cx, cy, r)

        if STOP_ON_PERFECT:
            is_perfect, _, _ = circle_perfect(img255, cx, cy, r, cos_tab, sin_tab)
            if is_perfect:
                l = metrics_loss_fn(cx, cy, r)
                if (l + IMPROVE_EPS) < best[0]:
                    best = (l, cx, cy, r)
                break

        l = metrics_loss_fn(cx, cy, r)
        if (l + IMPROVE_EPS) < best[0]:
            best = (l, cx, cy, r)
            if t >= WARMUP_STEPS: no_improve = 0
        else:
            if t >= WARMUP_STEPS:
                no_improve += 1
                if EARLY_STOP and (no_improve >= PATIENCE_STEPS): break

        if AUTO_EXTEND_STEPS and (max_steps < max_steps_cap):
            if mode in ("approach",) or (not white_growth_done and (outer_b > EPS_OUTER_EXPAND or (interior_black and inner_b < EPS_INNER_SHRINK))):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode in ("border_seek",):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)

        prev_mode = mode
        t += 1

    # --- final snap (optional, keeps proxy loss consistent) ---
    bx, by, br = best[1], best[2], best[3]
    (sx, sy, sr), _ = snap_refine_mask_iou(img255, bx, by, br, dxy=1, dr=2, thr=BLACK_THR, prefer_smaller_radius=True)
    snapped_loss = metrics_loss_fn(sx, sy, sr)
    if (snapped_loss <= best[0] + 1e-8):
        if return_initial:
            return snapped_loss, sx, sy, sr, initial_state
        return snapped_loss, sx, sy, sr

    if return_initial:
        return best[0], bx, by, br, initial_state
    return best

# ============================================================
# 10) Traced controller (GIF + IoU chase + overshoot tamers)
# ============================================================
def run_controller_trace(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, gt_tuple=None):
    size = img255.shape[0]
    cx, cy, r = choose_initial_state(nn, img255, img_small_bin, size, rng,
                                     policy=INIT_POLICY, use_init_head=USE_INIT_HEAD)

    trace = []
    trace.append({"t": -2, "cx": int(cx), "cy": int(cy), "r": int(r),
                  "loss": float('nan'), "mode": f"init({INIT_POLICY})"})

    best_loss = metrics_loss_fn(cx, cy, r)
    no_improve = 0
    trace.append({"t": -1, "cx": int(cx), "cy": int(cy), "r": int(r),
                  "loss": float(best_loss), "mode": "init_eval"})

    vx = 0.0; vy = 0.0
    scan_dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    scan_k = 0
    stuck_white = 0
    white_streak = 0
    prev_mode = "init"
    last_dist = None

    t = 0
    max_steps = int(steps)
    max_steps_cap = steps + EXTRA_STEPS_CAP_TRACE

    iou_best = -1.0
    iou_extra_used = 0

    while t < max_steps:
        x_in = build_input_vec(img_small_bin, cx, cy, r)
        out  = nn.forward(x_in)
        act, _ = split_outputs(out)
        sx, sy, move_step_nn, sr, rad_step_nn, _ = decode_actions(act, r)

        interior_black = any_black_interior(img255, cx, cy, r)
        border_dir = border_black_direction(img255, cx, cy, r, cos_tab, sin_tab)
        border_has_black = (border_dir is not None)
        all_white = (not interior_black) and (not border_has_black)

        if all_white:
            if sx == 0 and sy == 0:
                sx, sy = scan_dirs[scan_k % 4]; scan_k += 1
            if (sx > 0 and cx >= size-1-r) or (sx < 0 and cx <= r):   sx = -sx
            if (sy > 0 and cy >= size-1-r) or (sy < 0 and cy <= r):   sy = -sy
            if WHITE_USE_NN_STEP:
                move_frac = float(move_step_nn) / float(max(1, r))
                base_step = int(round(max(1.0, min(WHITE_STEP_ABS_MAX, move_frac * WHITE_STEP_ABS_MAX))))
            else:
                base_step = max(1, int(r))
            white_streak += 1
            grown = int(round(base_step * (1.0 + WHITE_STREAK_GROWTH * white_streak)))
            step  = int(min(WHITE_STEP_ABS_MAX, grown))
            jitter_cap = int(min(WHITE_JITTER_CLAMP, WHITE_JITTER_PX * white_streak))
            jx = int(rng.integers(-jitter_cap, jitter_cap + 1)) if jitter_cap > 0 else 0
            jy = int(rng.integers(-jitter_cap, jitter_cap + 1)) if jitter_cap > 0 else 0
            raw_dx = step * sx + jx
            raw_dy = step * sy + jy
            mode   = "white_scan"
            if white_streak % max(1, WHITE_SUPERJUMP_EVERY) == 0:
                tx = int(rng.integers(r, size - r)); ty = int(rng.integers(r, size - r))
                dx_f = float(tx - cx); dy_f = float(ty - cy); dist = math.hypot(dx_f, dy_f)
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    raw_dx = WHITE_SUPERJUMP_PX * ux + (rng.integers(-WHITE_JITTER_PX, WHITE_JITTER_PX + 1))
                    raw_dy = WHITE_SUPERJUMP_PX * uy + (rng.integers(-WHITE_JITTER_PX, WHITE_JITTER_PX + 1))
                    mode = "white_superjump"
            last_dist = None

        elif (not interior_black) and border_has_black:
            white_streak = 0
            step = int(max(1, min(BORDER_STEP_MAX_PX, int(r))))
            ux, uy = border_dir
            raw_dx = step * ux; raw_dy = step * uy
            mode = "border_seek"
            last_dist = None

        else:
            white_streak = 0
            cen = centroid_black_interior(img255, cx, cy, r)
            if cen is not None:
                tx, ty = cen
                dx_f = float(tx - cx); dy_f = float(ty - cy)
                dist = math.hypot(dx_f, dy_f)
                last_dist = dist
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    near = (dist < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
                    gain = (APPROACH_STEP_GAIN_NEAR if near else APPROACH_STEP_GAIN_FAR)
                    step = int(max(1, min(APPROACH_STEP_MAX_PX, round(dist * gain))))
                    raw_dx = step * ux; raw_dy = step * uy
                else:
                    raw_dx = 0.0; raw_dy = 0.0
            else:
                raw_dx = 0.0; raw_dy = 0.0
                last_dist = None
            mode = "approach"

        if mode in ("approach", "border_seek") and mode != prev_mode:
            vx *= MOMENTUM_DAMP_ON_MODE_CHANGE
            vy *= MOMENTUM_DAMP_ON_MODE_CHANGE

        if abs(raw_dx) <= DEADZONE_PX: raw_dx = 0.0
        if abs(raw_dy) <= DEADZONE_PX: raw_dy = 0.0
        vx = MOMENTUM_BETA * vx + (1.0 - MOMENTUM_BETA) * raw_dx
        vy = MOMENTUM_BETA * vy + (1.0 - MOMENTUM_BETA) * raw_dy
        dx = int(round(vx)); dy = int(round(vy))
        new_cx = cx + dx; new_cy = cy + dy
        new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)

        if mode.startswith("white_") and (new_cx == cx and new_cy == cy):
            stuck_white += 1
            if stuck_white >= WHITE_STUCK_PATIENCE:
                gx, gy = size // 2, size // 2
                dx_f = float(gx - cx); dy_f = float(gy - cy)
                dist = math.hypot(dx_f, dy_f)
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    vx = (1.0 - MOMENTUM_BETA) * WHITE_JUMP_PX * ux
                    vy = (1.0 - MOMENTUM_BETA) * WHITE_JUMP_PX * uy
                    dx = int(round(vx)); dy = int(round(vy))
                    new_cx = cx + dx; new_cy = cy + dy
                    new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        # ---------------- WHITE-SCAN RADIUS GROWTH (NEW) ----------------
        white_growth_done = False
        if all_white:
            grow_ready = (white_streak >= WHITE_RADIUS_GROW_AFTER)
            if grow_ready:
                periodic_ok = ((white_streak - WHITE_RADIUS_GROW_AFTER) % max(1, WHITE_RADIUS_GROW_EVERY) == 0)
                bit_ok = (sr > 0) if WHITE_RADIUS_REQUIRE_BIT else True
                if periodic_ok and bit_ok:
                    raw = abs(nn_radius_delta(sr, rad_step_nn))
                    grow_px = int(min(max(RAD_STEP_FLOOR_PX, raw), WHITE_RADIUS_GROW_MAX_PX))
                    if grow_px > 0:
                        cx, cy, r = apply_radius_recenter(size, cx, cy, r, +grow_px)
                        cx, cy, r = enforce_bounds(size, cx, cy, r)
                        if WHITE_RADIUS_RESET_STREAK:
                            white_streak = 0
                        white_growth_done = True
                        mode_r = "white_grow"
                    else:
                        mode_r = None
                else:
                    mode_r = None
            else:
                mode_r = None
        else:
            mode_r = None
        # ----------------------------------------------------------------

        # radius control with distance-aware caps (skip if we already grew)
        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)
        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if outer_b > EPS_OUTER_EXPAND:
                raw = abs(nn_radius_delta(sr, rad_step_nn))
                delta_r = +min(raw, cap_px)
                cx, cy, r = apply_radius_recenter(size, cx, cy, r, delta_r)
                mode_r = "expand_nn"
            elif interior_black and (inner_b < EPS_INNER_SHRINK):
                raw = abs(nn_radius_delta(sr, rad_step_nn))
                delta_r = -min(raw, cap_px)
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
                mode_r = "shrink_nn"

        cx, cy, r = enforce_bounds(size, cx, cy, r)

        if STOP_ON_PERFECT:
            is_perfect, _, _ = circle_perfect(img255, cx, cy, r, cos_tab, sin_tab)
            if is_perfect:
                l = metrics_loss_fn(cx, cy, r)
                trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                              "loss": float(l), "mode": "perfect_stop"})
                break

        if gt_tuple is not None:
            xg, yg, rg = gt_tuple

            exact_match = (int(cx) == int(xg) and int(cy) == int(yg) and int(r) == int(rg))
            if IOU_EVAL_STOP and exact_match:
                l = metrics_loss_fn(cx, cy, r)
                trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                              "loss": float(l), "mode": "iou_stop_equal"})
                break

            iou_now = iou_circle(size, (cx, cy, r), (xg, yg, rg))
            if IOU_EVAL_STOP and (iou_now >= IOU_EVAL_THRESH):
                l = metrics_loss_fn(cx, cy, r)
                trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                              "loss": float(l), "mode": "iou_stop"})
                break

            if IOU_CHASE_ENABLE and (iou_now >= iou_best + IOU_IMPROVE_DELTA) and (iou_extra_used < IOU_EXTRA_CAP):
                add = min(IOU_STEPS_BONUS, IOU_EXTRA_CAP - iou_extra_used)
                max_steps = min(max_steps_cap, max_steps + add)
                iou_extra_used += add
                iou_best = iou_now

        l = metrics_loss_fn(cx, cy, r)
        trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                      "loss": float(l), "mode": mode if mode_r is None else mode_r})

        if (l + IMPROVE_EPS) < best_loss:
            best_loss = l
            if t >= WARMUP_STEPS: no_improve = 0
        else:
            if t >= WARMUP_STEPS:
                no_improve += 1
                if EARLY_STOP and (no_improve >= PATIENCE_STEPS): break

        if AUTO_EXTEND_STEPS and (max_steps < max_steps_cap):
            if mode in ("approach",) or (mode_r in ("shrink_nn","expand_nn","white_grow")):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode in ("border_seek",):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)

        prev_mode = mode
        t += 1

    good = [p for p in trace if not (isinstance(p["loss"], float) and math.isnan(p["loss"]))]
    best_idx = int(np.argmin([p["loss"] for p in good])) if good else 0
    best = good[best_idx] if good else trace[-1]

    (snap_cx, snap_cy, snap_r), snap_iou = snap_refine_mask_iou(
        img255, best["cx"], best["cy"], best["r"],
        dxy=1, dr=2, thr=BLACK_THR, prefer_smaller_radius=True
    )

    if (snap_cx, snap_cy, snap_r) != (best["cx"], best["cy"], best["r"]):
        snapped_loss = metrics_loss_fn(snap_cx, snap_cy, snap_r)
        if (snapped_loss <= best["loss"] + 1e-8):
            best = {
                "t": best["t"],
                "cx": int(snap_cx), "cy": int(snap_cy), "r": int(snap_r),
                "loss": float(snapped_loss),
                "mode": "snap_refine"
            }
            trace.append({"t": best["t"], "cx": best["cx"], "cy": best["cy"],
                          "r": best["r"], "loss": float(snapped_loss),
                          "mode": "snap_refine"})

    return trace, best

# ============================================================
# 11) Drawing & GIF
# ============================================================
def _draw_frame(img255, p, gt=None, scale=GIF_SCALE):
    cx, cy, r = p["cx"], p["cy"], p["r"]
    t, loss, mode = p["t"], p["loss"], p.get("mode","")

    # Base frame
    base = Image.fromarray(img255).convert("RGB")
    if scale != 1:
        base = base.resize((img255.shape[1]*scale, img255.shape[0]*scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)

    # Bounding boxes
    bbox_pred = [int((cx - r) * scale), int((cy - r) * scale),
                 int((cx + r) * scale), int((cy + r) * scale)]

    bbox_gt = None
    if gt is not None:
        gx, gy, gr = gt
        bbox_gt = [int((gx - gr) * scale), int((gy - gr) * scale),
                   int((gx + gr) * scale), int((gy + gr) * scale)]

    # Line widths: make green a bit thinner so red dominates at exact overlap
    w_pred = max(1, scale)
    w_gt   = max(1, w_pred - 1)

    # Colors
    RED   = (255, 0, 0)
    GREEN = (0, 200, 0)

    # Draw order: green first, red last (if enabled via PRED_ON_TOP flag)
    if 'PRED_ON_TOP' in globals() and PRED_ON_TOP:
        if SHOW_GT_IN_GIF and (bbox_gt is not None):
            draw.ellipse(bbox_gt, outline=GREEN, width=w_gt)
        draw.ellipse(bbox_pred, outline=RED, width=w_pred)
    else:
        draw.ellipse(bbox_pred, outline=RED, width=w_pred)
        if SHOW_GT_IN_GIF and (bbox_gt is not None):
            draw.ellipse(bbox_gt, outline=GREEN, width=w_gt)

    # -------- HUD text (configurable color + outline + optional background) --------
    # Font (scaled by GIF scale)
    font = None
    try:
        if HUD_FONT_PATH:
            font = ImageFont.truetype(HUD_FONT_PATH, size=max(8, int(HUD_FONT_SIZE * scale)))
    except Exception:
        font = None
    if font is None:
        # default PIL font (bitmap), looks fine when scaled as we render on a resized image
        font = ImageFont.load_default()

    # Text content
    loss_txt = f"{loss:.4f}" if isinstance(loss, (float,int)) and not math.isnan(loss) else "NA"
    hud_text = f"t={t} loss={loss_txt} ({cx},{cy},r={r}) mode={mode}"

    # Position
    x0 = int(HUD_POS[0] * scale)
    y0 = int(HUD_POS[1] * scale)

    # Stroke width scales with the image scale
    sw = max(1, scale // 2)

    # Optional translucent background box for maximum readability
    if HUD_BG_RGBA is not None:
        # Compute text bbox using a temp drawer that supports textbbox
        tmp_draw = ImageDraw.Draw(base)
        # textbbox accounts for stroke if stroke_width passed
        left, top, right, bottom = tmp_draw.textbbox((x0, y0), hud_text, font=font, stroke_width=sw)
        pad = int(HUD_PAD * scale)
        rect = (left - pad, top - pad, right + pad, bottom + pad)

        # Draw on an RGBA overlay to keep background semi-transparent
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle(rect, fill=HUD_BG_RGBA)
        # Draw the text with outline on the overlay (alpha-safe)
        odraw.text((x0, y0), hud_text,
                   font=font,
                   fill=HUD_TEXT_COLOR + ((255,) if len(HUD_TEXT_COLOR) == 3 else ()),
                   stroke_width=sw,
                   stroke_fill=HUD_STROKE_COLOR + ((255,) if len(HUD_STROKE_COLOR) == 3 else ()))
        # Composite and convert back to RGB
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    else:
        # No background box, just text with outline directly on the frame
        draw.text((x0, y0), hud_text,
                  font=font,
                  fill=HUD_TEXT_COLOR,
                  stroke_width=sw,
                  stroke_fill=HUD_STROKE_COLOR)

    return base

def save_gif_for_trace(img255, trace, gt_tuple, out_path):
    # Trim the trace if requested, so GIF stops exactly when the controller stopped
    if 'TRIM_GIF_AT_STOP' in globals() and TRIM_GIF_AT_STOP:
        trace = trim_trace_on_stop(trace, keep_snap=KEEP_SNAP_AFTER_STOP)

    gt_draw = gt_tuple if SHOW_GT_IN_GIF else None
    frames = [_draw_frame(img255, p, gt_draw) for p in trace]

    # Ensure at least 2 frames (GIF encoders can be picky with single-frame animations)
    if len(frames) == 1:
        frames = frames * 2

    # Base durations: all frames use the standard step duration
    durations = [int(GIF_DURATION_MS)] * len(frames)

    # Extend visibility of the LAST frame
    if GIF_TAIL_COMPAT_DUPLICATE:
        # Compatibility mode: replace the single long last delay by repeatedly appending
        # the last frame so total visible time ~= GIF_TAIL_HOLD_MS.
        hold_ms = max(0, int(GIF_TAIL_HOLD_MS))
        if hold_ms > 0:
            # Remove current last frame (we'll re-add it duplicated)
            last = frames[-1]
            frames = frames[:-1]
            durations = durations[:-1]

            # How many duplicates?
            n = max(1, int(math.ceil(hold_ms / float(GIF_TAIL_DUPLICATE_EACH_MS))))
            frames.extend([last] * n)
            durations.extend([int(GIF_TAIL_DUPLICATE_EACH_MS)] * n)
    else:
        # Simple mode: make just the last frame hold longer
        if len(durations) > 0:
            durations[-1] = int(GIF_TAIL_HOLD_MS)

    # Write GIF with per-frame durations
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,       # <-- per-frame durations
        loop=0,
        optimize=False,
        disposal=2
    )

# ============================================================
# Trace post-processing: trim at first stop + consistent steps_used
# ============================================================
STOP_MODES_ORDERED = ["iou_stop_equal", "iou_stop", "perfect_stop"]

def _first_stop_index(trace):
    """
    Return (idx, mode) for the first stop event in the preferred order.
    If multiple stops appear, the earliest in time is chosen; if same time,
    priority is iou_stop_equal > iou_stop > perfect_stop.
    """
    best = None  # (t, idx, priority)
    pri = {m:i for i,m in enumerate(STOP_MODES_ORDERED)}
    for idx, p in enumerate(trace):
        m = p.get("mode", "")
        if m in pri and isinstance(p.get("t"), int):
            t = p["t"]
            cand = (t, idx, pri[m], m)
            if best is None or t < best[0] or (t == best[0] and pri[m] < best[2]):
                best = cand
    if best is None:
        return None, None
    return best[1], best[3]

def trim_trace_on_stop(trace, keep_snap=KEEP_SNAP_AFTER_STOP):
    """
    Slice the trace up to and including the first stop frame.
    Optionally keep a 'snap_refine' right after it if it shares the same t.
    """
    idx, _ = _first_stop_index(trace)
    if idx is None:
        return trace
    end = idx
    if keep_snap and (idx + 1) < len(trace):
        nxt = trace[idx + 1]
        if nxt.get("mode") == "snap_refine" and nxt.get("t") == trace[idx].get("t"):
            end = idx + 1
    return trace[:end + 1]

def steps_used_from_trace(trace):
    """
    Use the *trimmed* trace semantics: if a stop exists, steps_used = stop_t + 1,
    else use last t + 1 when budget/exhaustion.
    """
    idx, _ = _first_stop_index(trace)
    if idx is not None and isinstance(trace[idx].get("t"), int):
        return trace[idx]["t"] + 1
    # fallback: last integer t in the trace
    t_last = None
    for p in trace:
        if isinstance(p.get("t"), int):
            t_last = p["t"]
    return (t_last + 1) if t_last is not None else None

# ============================================================
# 12) Net config / persistence
# ============================================================
input_size   = IN_SIZE * IN_SIZE + 3
hidden_sizes = [6]
output_size  = ACTION_BITS + (3 if USE_INIT_HEAD else 0)
nn = NeuralNetwork(input_size, hidden_sizes, output_size, use_init_head=USE_INIT_HEAD)

CHECKPOINT_TAG   = f"v12_overshoot_tamed_iouchase_borderaware_initpolicy_{INIT_POLICY}_inithead_{int(USE_INIT_HEAD)}"
CHECKPOINT_DIR   = "checkpoints"
CHECKPOINT_PATH  = os.path.join(CHECKPOINT_DIR, f"ckpt_single_ga_controller_{CHECKPOINT_TAG}.npz")

def _arch_signature(num_weights, num_bits):
    return {
        "IN_SIZE": IN_SIZE,
        "IMG_SIZE": IMG_SIZE,
        "R_NORM": R_NORM,
        "hidden_sizes": hidden_sizes.copy(),
        "output_size": output_size,
        "num_weights": int(num_weights),
        "num_bits": int(num_bits),
        "tag": CHECKPOINT_TAG,
        "version": f"2025-10-30.v12_overshoot_tamed",
    }

def save_checkpoint(nn, popx, num_weights, num_bits, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    weights = nn.get_weights().astype(np.float32)
    arch = _arch_signature(num_weights, num_bits)
    meta_json = json.dumps(arch)
    np.savez_compressed(path, weights=weights, pop=popx.astype(np.uint8), meta=meta_json)

def load_checkpoint(expected_num_weights, expected_num_bits, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return None, None
    try:
        data = np.load(path, allow_pickle=False)
        weights = data.get("weights", None)
        pop = data.get("pop", None)
        meta_json = data.get("meta", None)
        w_ok = (weights is not None) and (weights.size == expected_num_weights)
        p_ok = (pop is not None) and (pop.ndim == 2) and (pop.shape[1] == expected_num_bits)
        if meta_json is not None:
            try:
                meta = json.loads(str(meta_json))
                if meta.get("tag") != CHECKPOINT_TAG: return None, None
                if meta.get("num_weights") != expected_num_weights or meta.get("num_bits") != expected_num_bits:
                    return None, None
            except Exception:
                return None, None
        return (weights if w_ok else None), (pop if p_ok else None)
    except Exception:
        return None, None

# ============================================================
# 13) Load annotations
# ============================================================
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]
annotations = [rec for rec in records if rec.get("split") == "single"]

# ============================================================
# 14) GA loop + final execution (with GIFs)
# ============================================================
initial_population = None
_num_weights = nn.get_weights().size
_num_bits    = int(_num_weights * 16)

ckpt_weights, ckpt_pop = load_checkpoint(_num_weights, _num_bits, CHECKPOINT_PATH)
if ckpt_weights is not None:
    nn.set_weights(ckpt_weights.astype(np.float32))
    print(f"[ckpt] Weights loaded: {CHECKPOINT_PATH} (num_weights={_num_weights})")
if ckpt_pop is not None:
    initial_population = ckpt_pop.astype(np.uint8)
    print(f"[ckpt] GA population loaded: {initial_population.shape}")

sum_iou = 0.0
cnt = 0
cnt_iou_good = 0
cnt_stuck = 0

RUNS_DIR and os.path.exists(RUNS_DIR)
logf = open(RUN_JSONL_PATH, 'w', encoding='utf-8')
gif_count = 0

for ann in annotations:
    file_rel = ann["file"]
    file_path = os.path.join(DATA_ROOT, file_rel)
    img_small = load_image_small_bin(file_path, out_size=IN_SIZE)
    img_full  = load_image_full_gray(file_path)

    probe_list_coarse = _make_probe_list(PROBE_R_COARSE, img_full.shape[0])
    probe_list_fine   = _make_probe_list(PROBE_R_FINE,   img_full.shape[0])

    cache_coarse = {}
    cache_fine   = {}
    coarse_weights = (0.7, 1.0, 0.9, 0.05)
    fine_weights   = (0.8, 1.1, 1.0, 0.10)

    metrics_loss_coarse = make_metrics_loss(
        img_full, COS_COARSE, SIN_COARSE, cache_coarse, probe_list_coarse,
        weights=coarse_weights, w_probe=W_PROBE_COARSE
    )
    metrics_loss_fine   = make_metrics_loss(
        img_full, COS_FINE,   SIN_FINE,   cache_fine,   probe_list_fine,
        weights=fine_weights, w_probe=W_PROBE_FINE
    )

    circle = ann["circles"][0]
    x_real, y_real, r_real = int(circle["cx"]), int(circle["cy"]), int(circle["r"])

    base_loss, cx0, cy0, r0, initial_state = run_controller(
        nn, img_full, img_small,
        steps=0,
        cos_tab=COS_COARSE, sin_tab=SIN_COARSE,
        metrics_loss_fn=metrics_loss_coarse,
        probe_r_list=probe_list_coarse,
        return_initial=True
    )
    cx_init, cy_init, r_init, loss_init_true = initial_state

    def fit_func_coarse(bits):
        w = bits2bytes(bits, 'int16').astype(np.float32) / 1000.0
        nn.set_weights(w)
        loss, _, _, _ = run_controller(
            nn, img_full, img_small,
            steps=CTRL_STEPS_COARSE,
            cos_tab=COS_COARSE, sin_tab=SIN_COARSE,
            metrics_loss_fn=metrics_loss_coarse,
            probe_r_list=probe_list_coarse
        )
        return loss

    popsize_stage1 = initial_population.shape[0] if (initial_population is not None) else GA_POP_COARSE
    gaoptions1 = {
        "PopulationSize": popsize_stage1,
        "Generations": GA_GENS_COARSE,
        "InitialPopulation": initial_population,
        "MutationFcn": GA_MUT_COARSE,
        "EliteCount": ELITE_COUNT,
    }
    x_best, popx, fitvals = gago(fit_func_coarse, _num_bits, gaoptions1)

    if loss_init_true <= 0.6:
        steps_fine = max(10, CTRL_STEPS_FINE - 2)
    elif loss_init_true <= 1.2:
        steps_fine = CTRL_STEPS_FINE
    else:
        steps_fine = max(CTRL_STEPS_FINE, 16)

    def fit_func_fine(bits):
        w = bits2bytes(bits, 'int16').astype(np.float32) / 1000.0
        nn.set_weights(w)
        loss, _, _, _ = run_controller(
            nn, img_full, img_small,
            steps=steps_fine,
            cos_tab=COS_FINE, sin_tab=SIN_FINE,
            metrics_loss_fn=metrics_loss_fine,
            probe_r_list=probe_list_fine
        )
        return loss

    popsize_stage2 = popx.shape[0]
    gaoptions2 = {
        "PopulationSize": popsize_stage2,
        "Generations": GA_GENS_POLISH,
        "InitialPopulation": popx,
        "MutationFcn": GA_MUT_POLISH,
        "EliteCount": ELITE_COUNT,
    }
    x_best, popx, fitvals = gago(fit_func_fine, _num_bits, gaoptions2)

    best_weights = bits2bytes(x_best, 'int16').astype(np.float32) / 1000.0
    nn.set_weights(best_weights)

    initial_population = popx
    save_checkpoint(nn, initial_population, _num_weights, _num_bits, CHECKPOINT_PATH)

    trace, best = run_controller_trace(
        nn, img_full, img_small,
        steps=int(steps_fine * max(1, INFER_STEPS_MULT)),
        cos_tab=COS_FINE, sin_tab=SIN_FINE,
        metrics_loss_fn=metrics_loss_fine,
        gt_tuple=(x_real, y_real, r_real)
    )
    final_loss = float(best["loss"])
    cx_pred, cy_pred, r_pred = int(best["cx"]), int(best["cy"]), int(best["r"])

    fill     = interior_fill_fraction(img_full, cx_pred, cy_pred, r_pred)
    inner_b  = _ring_fraction_vec(img_full, cx_pred, cy_pred, r_pred, delta=-1, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    outer_b  = _ring_fraction_vec(img_full, cx_pred, cy_pred, r_pred, delta=+1, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    cut      = _border_cut_vec(img_full, cx_pred, cy_pred, r_pred, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    probe_b  = _probe_max_thick(img_full, cx_pred, cy_pred,
                                _make_probe_list(PROBE_R_FINE, IMG_SIZE),
                                PROBE_THICKNESS, COS_FINE, SIN_FINE)
    iou      = iou_circle(IMG_SIZE, (cx_pred, cy_pred, r_pred), (x_real, y_real, r_real))

    cnt += 1; sum_iou += iou
    if iou >= 0.5: cnt_iou_good += 1
    if final_loss >= 1.99: cnt_stuck += 1

    gif_path = None
    if MAKE_GIFS and (GIF_LIMIT is None or gif_count < GIF_LIMIT):
        safe_name = file_rel.replace("/", "__")
        gif_path = os.path.join(GIFS_DIR, f"{os.path.splitext(safe_name)[0]}_{RUN_ID}.gif")
        save_gif_for_trace(img_full, trace, (x_real, y_real, r_real), gif_path)
        gif_count += 1
        print(f"[gif] salvo: {gif_path}")

    # Determine stop reason in consistent priority, and compute steps_used reliably
    _, stop_mode_found = _first_stop_index(trace)
    stop_reason = stop_mode_found if stop_mode_found is not None else "budget"
    steps_used = steps_used_from_trace(trace)

    print(f"Imagem: {file_path}")
    print(f"GT (px):        (x={x_real}, y={y_real}, r={r_real})")
    print(f"Inicial (px):   (x={cx_init}, y={cy_init}, r={r_init}), loss_inicial={loss_init_true:.6f}")
    print(f"Predição (px):  (x={cx_pred}, y={cy_pred}, r={r_pred})")
    print(f"Loss final:     {final_loss:.6f} | fill={fill:.4f} inner={inner_b:.4f} outer={outer_b:.4f} cut={cut:.4f} probe={probe_b:.4f} IoU={iou:.4f}")
    print(f"Stop reason:    {stop_reason} | steps_used={steps_used}")
    print("--------")

    log_line = {
        "run_id": RUN_ID,
        "file": file_rel,
        "gt": {"x": x_real, "y": y_real, "r": r_real},
        "init": {"x": cx_init, "y": cy_init, "r": r_init, "loss": float(loss_init_true)},
        "pred": {"x": cx_pred, "y": cy_pred, "r": r_pred, "loss": float(final_loss)},
        "metrics": {"fill": float(fill), "inner": float(inner_b), "outer": float(outer_b),
                    "cut": float(cut), "probe": float(probe_b), "IoU": float(iou)},
        "ctrl": {"coarse_steps": CTRL_STEPS_COARSE, "fine_steps": steps_fine,
                 "momentum_beta": MOMENTUM_BETA, "use_gray_code": USE_GRAY_CODE,
                 "policy": "white_scan | white_superjump | border_seek | approach | shrink_nn | expand_nn | perfect_stop | iou_stop",
                 "auto_extend": AUTO_EXTEND_STEPS,
                 "init_policy": INIT_POLICY, "use_init_head": USE_INIT_HEAD, "init_nn_steps": INIT_NN_STEPS,
                 "search_budget": SEARCH_BUDGET, "early_stop": EARLY_STOP,
                 "infer_steps_mult": INFER_STEPS_MULT},
        "stop": {"reason": stop_reason, "steps_used": steps_used},
        "gif": gif_path, "time": time.time()
    }
    logf.write(json.dumps(log_line, ensure_ascii=False) + "\n")
    logf.flush()

# dataset summary
if cnt > 0:
    mean_iou  = sum_iou / float(cnt)
    pct_good  = 100.0 * cnt_iou_good / float(cnt)
    pct_stuck = 100.0 * cnt_stuck / float(cnt)
    print(f"[Resumo dataset] imagens={cnt} | IoU médio={mean_iou:.3f} | %IoU≥0.5={pct_good:.1f}% | %stuck≈2.0={pct_stuck:.1f}%")
    summary = {
        "run_id": RUN_ID,
        "dataset_images": cnt,
        "mean_IoU": float(mean_iou),
        "pct_IoU_ge_0_5": float(pct_good),
        "pct_stuck_ge_approx_2": float(pct_stuck),
        "config": {
            "IN_SIZE": IN_SIZE, "IMG_SIZE": IMG_SIZE, "R_NORM": R_NORM,
            "RING_SAMPLES_COARSE": RING_SAMPLES_COARSE, "RING_SAMPLES_FINE": RING_SAMPLES_FINE,
            "W_PROBE_COARSE": W_PROBE_COARSE, "W_PROBE_FINE": W_PROBE_FINE,
            "GA_POP": GA_POP_COARSE, "GA_GENS_COARSE": GA_GENS_COARSE, "GA_GENS_POLISH": GA_GENS_POLISH,
            "GA_MUT_COARSE": GA_MUT_COARSE, "GA_MUT_POLISH": GA_MUT_POLISH,
            "MOMENTUM_BETA": MOMENTUM_BETA, "USE_GRAY_CODE": USE_GRAY_CODE,
            "MAKE_GIFS": MAKE_GIFS, "GIF_SCALE": GIF_SCALE, "GIF_DURATION_MS": GIF_DURATION_MS,
            "POLICY": "white_scan | white_superjump | border_seek | approach | shrink_nn | expand_nn | perfect_stop | iou_stop",
            "AUTO_EXTEND_STEPS": AUTO_EXTEND_STEPS,
            "EXTRA_STEPS_CAP": EXTRA_STEPS_CAP,
            "EXTRA_STEPS_CAP_TRACE": EXTRA_STEPS_CAP_TRACE,
            "SEARCH_BUDGET": SEARCH_BUDGET, "INFER_STEPS_MULT": INFER_STEPS_MULT,
            "BLACK_THR": BLACK_THR,
            "STOP_ON_PERFECT": STOP_ON_PERFECT,
            "PERFECT_INNER_FRAC": PERFECT_INNER_FRAC,
            "PERFECT_OUTER_FRAC": PERFECT_OUTER_FRAC,
            "PERFECT_THICKNESS": PERFECT_THICKNESS,
            "WHITE_STREAK_GROWTH": WHITE_STREAK_GROWTH,
            "WHITE_JITTER_PX": WHITE_JITTER_PX,
            "WHITE_JITTER_CLAMP": WHITE_JITTER_CLAMP,
            "WHITE_SUPERJUMP_EVERY": WHITE_SUPERJUMP_EVERY,
            "WHITE_SUPERJUMP_PX": WHITE_SUPERJUMP_PX,
            "APPROACH_STEP_GAIN_FAR": APPROACH_STEP_GAIN_FAR,
            "APPROACH_STEP_GAIN_NEAR": APPROACH_STEP_GAIN_NEAR,
            "APPROACH_NEAR_FRAC_OF_R": APPROACH_NEAR_FRAC_OF_R,
            "APPROACH_STEP_MAX_PX": APPROACH_STEP_MAX_PX,
            "BORDER_STEP_MAX_PX": BORDER_STEP_MAX_PX,
            "MOMENTUM_DAMP_ON_MODE_CHANGE": MOMENTUM_DAMP_ON_MODE_CHANGE,
            "EPS_OUTER_EXPAND": EPS_OUTER_EXPAND,
            "EPS_INNER_SHRINK": EPS_INNER_SHRINK,
            "RAD_CAP_NEAR_PX": RAD_CAP_NEAR_PX,
            "RAD_CAP_FAR_PX": RAD_CAP_FAR_PX,
            "IOU_EVAL_STOP": IOU_EVAL_STOP,
            "IOU_EVAL_THRESH": IOU_EVAL_THRESH,
            "IOU_CHASE_ENABLE": IOU_CHASE_ENABLE,
            "IOU_IMPROVE_DELTA": IOU_IMPROVE_DELTA,
            "IOU_STEPS_BONUS": IOU_STEPS_BONUS,
            "IOU_EXTRA_CAP": IOU_EXTRA_CAP
        }
    }
    with open(RUN_JSONL_PATH, 'a', encoding='utf-8') as fsum:
        fsum.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(f"[Logs] JSONL salvo em: {RUN_JSONL_PATH}")
