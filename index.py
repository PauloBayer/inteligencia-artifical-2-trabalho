# index.py
# ------------------------------------------------------------
# Controlador binário (1 bola) — NN decide a MAGNITUDE do ajuste de raio
# ------------------------------------------------------------
# Modos:
#  - white_scan: varredura quando tudo é branco
#  - border_seek: desloca 1*raio na direção média dos pretos de borda
#  - approach: aproxima do centróide quando há preto no interior
#  - shrink_nn: alvo contido e pequeno → NN decide quanto DIMINUIR r
#  - expand_nn: dentro da bola cortando borda → NN decide quanto AUMENTAR r
#  - refine_nn: círculo completo → NN decide pequeno ajuste (±), limitado
# Limites rígidos: círculo SEMPRE dentro de 255×255 (enforce_bounds)
# GA, Gray code, momentum, GIFs e logs preservados.
# ------------------------------------------------------------

import os
import json
import time
import math
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

# GA + utils
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

# ============================================================
# 1) Hiperparâmetros
# ============================================================
CTRL_STEPS_COARSE = 8
CTRL_STEPS_FINE   = 12
PATIENCE_STEPS    = 4
WARMUP_STEPS      = 3
IMPROVE_EPS       = 1e-6

STEP_FLOOR_PX_BASE = 1
STEP_FLOOR_PX_MAX  = 3

RING_SAMPLES_COARSE = 128
RING_SAMPLES_FINE   = 256

PROBE_R_COARSE   = 48
PROBE_R_FINE     = 64
PROBE_THICKNESS  = 2
W_PROBE_COARSE   = 0.35
W_PROBE_FINE     = 0.10

# Critérios de “círculo completo”
TH_INNER = 0.90
TH_OUTER = 0.10

# >>> Ajuste de raio governado pela NN <<<
RAD_STEP_FLOOR_PX     = 1    # piso do |Δr| vindo da NN
COMPLETE_REFINE_MAX   = 3    # teto de |Δr| em refine_nn (para não oscilar)

# >>> Critérios de expansão (o "quando") <<<
TH_EXPAND_INNER = 0.85   # interior (r-1) bem preto → estamos dentro da bola
TH_EXPAND_OUTER = 0.50   # borda (r+1) ainda muito preta → ainda cortando bola

# >>> Steps extras automáticos (dão mais tempo à rede) <<<
AUTO_EXTEND_STEPS       = True
EXTRA_STEPS_HAS_SIGNAL  = 8   # approach / shrink_nn / expand_nn sem estar “completo”
EXTRA_STEPS_BORDER      = 6   # border_seek
EXTRA_STEPS_REFINE      = 4   # refine_nn
EXTRA_STEPS_CAP         = 24  # teto adicional total

# GA
GA_POP_COARSE   = 120
GA_GENS_COARSE  = 22
GA_MUT_COARSE   = 0.90
GA_POP_POLISH   = GA_POP_COARSE
GA_GENS_POLISH  = 6
GA_MUT_POLISH   = 0.90
ELITE_COUNT     = 2

# Checkpoint
CHECKPOINT_DIR   = "checkpoints"
CHECKPOINT_TAG   = "v7_nn_controls_radius_amount_bounded"
CHECKPOINT_PATH  = os.path.join(CHECKPOINT_DIR, f"ckpt_single_ga_controller_{CHECKPOINT_TAG}.npz")

# Decodificação (rede)
USE_GRAY_CODE     = True
MOVE_GAMMA        = 0.85
RAD_GAMMA         = 0.85

# Movimento
MOMENTUM_BETA     = 0.6
DEADZONE_PX       = 0

# Logs/GIFs
RUNS_DIR          = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_ID            = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_JSONL_PATH    = os.path.join(RUNS_DIR, f"run_{RUN_ID}.jsonl")

MAKE_GIFS         = True
GIFS_DIR          = os.path.join(RUNS_DIR, "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)
GIF_SCALE         = 2
GIF_DURATION_MS   = 120
GIF_LIMIT         = None  # limite opcional

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
# 3) Amostragem / métricas
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
    black = np.count_nonzero(vals == 0)
    total = int(xs.size)
    return black / float(total)

def _ring_fraction_thick(img255, cx, cy, r, delta_center, thickness, cos_tab, sin_tab):
    acc = 0.0
    cnt = 0
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
    filled = int(np.count_nonzero(img255[mask] == 0))
    return filled / float(area)

def iou_circle(size, c1, c2):
    m1 = _circle_mask(size, c1[0], c1[1], c1[2])
    m2 = _circle_mask(size, c2[0], c2[1], c2[2])
    inter = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    return (inter / union) if union > 0 else 0.0

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
# 5) Rede (tiny MLP)
# ============================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
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

# ============================================================
# 6) Decodificação (22 bits) c/ Gray code
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
    bits = (out_vec > 0.5).astype(np.uint8)
    bx_pos, bx_neg, by_pos, by_neg = map(int, bits[0:4])
    sx = 1 if (bx_pos and not bx_neg) else (-1 if (bx_neg and not bx_pos) else 0)
    sy = 1 if (by_pos and not by_neg) else (-1 if (by_neg and not by_pos) else 0)
    k_move_u8 = _decode_u8(bits[4:12], use_gray=USE_GRAY_CODE)
    move_frac = _smooth_frac(k_move_u8, gamma=MOVE_GAMMA)
    move_step = float(move_frac) * float(max(1, r_curr))  # usado em scan
    br_up, br_down = int(bits[12]), int(bits[13])
    sr = 1 if (br_up and not br_down) else (-1 if (br_down and not br_up) else 0)
    k_rad_u8 = _decode_u8(bits[14:22], use_gray=USE_GRAY_CODE)
    rad_frac = _smooth_frac(k_rad_u8, gamma=RAD_GAMMA)
    rad_step = float(rad_frac) * float(max(1, r_curr))  # [0..r_curr]
    return sx, sy, move_step, sr, rad_step, bits

# ============================================================
# 7) Estado / inicialização / limites
# ============================================================
def build_input_vec(img_small_bin, cx, cy, r):
    state = np.array([cx/IMG_SIZE, cy/IMG_SIZE, r/float(R_NORM)], dtype=np.float32)
    return np.concatenate([img_small_bin.flatten(), state], axis=0).astype(np.float32)

def r_fit_for_center(size, cx, cy):
    """Maior raio que mantém o círculo totalmente dentro da imagem."""
    return int(max(0, min(cx, cy, size-1-cx, size-1-cy)))

def initial_center_fit_all(size):
    cx = size // 2; cy = size // 2
    r  = r_fit_for_center(size, cx, cy)  # ≈127 em 255x255
    return cx, cy, r

def clamp_center_with_radius(size, cx, cy, r):
    """Garante cx,cy em [r, size-1-r]."""
    cx = int(np.clip(cx, r, size-1-r))
    cy = int(np.clip(cy, r, size-1-r))
    return cx, cy

def enforce_bounds(size, cx, cy, r):
    """Aplica: r ≤ r_fit(cx,cy) e cx,cy ∈ [r, size-1-r]."""
    rmax = r_fit_for_center(size, cx, cy)
    r = int(min(r, rmax))
    cx, cy = clamp_center_with_radius(size, cx, cy, r)
    return cx, cy, r

# ============================================================
# 8) Heurísticas auxiliares
# ============================================================
def any_black_interior(img255, cx, cy, r):
    mask = _circle_mask(img255.shape[0], cx, cy, r)
    if not np.any(mask): return False
    return np.any(img255[mask] == 0)

def border_black_direction(img255, cx, cy, r, cos_tab, sin_tab):
    """Direção média (ux,uy) dos pixels pretos na borda (delta=0)."""
    size = img255.shape[0]
    rr = int(max(1, abs(int(round(r)))))
    xs, ys = _ring_coords(cx, cy, rr, cos_tab, sin_tab, size)
    if xs.size == 0: return None
    mask_black = (img255[ys, xs] == 0)
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
    ys, xs = np.where(mask & (img255 == 0))
    if xs.size == 0: return None
    mx = int(np.clip(int(np.rint(xs.mean())), 0, size - 1))
    my = int(np.clip(int(np.rint(ys.mean())), 0, size - 1))
    return (mx, my)

# ============================================================
# 9) Helpers de ajuste de raio guiado pela NN
# ============================================================
def nn_radius_delta(sr, rad_step):
    """Converte (direção, magnitude) em Δr inteiro com piso."""
    step = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step))))
    return int(sr) * step

def apply_radius(size, cx, cy, r, delta_r):
    """Aplica Δr e reforça limites."""
    r_new = max(1, r + int(delta_r))
    cx, cy, r_new = enforce_bounds(size, cx, cy, r_new)
    return cx, cy, r_new

# ============================================================
# 10) Controlador (sem tracing) — NN comanda magnitude do raio
# ============================================================
def run_controller(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, probe_r_list,
                   return_initial=False):
    size = img255.shape[0]
    cx, cy, r = initial_center_fit_all(size)
    cx, cy, r = enforce_bounds(size, cx, cy, r)

    initial_loss = metrics_loss_fn(cx, cy, r)
    initial_state = (cx, cy, r, initial_loss)

    best = (initial_loss, cx, cy, r)
    no_improve = 0

    vx = 0.0; vy = 0.0
    scan_dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    scan_k = 0

    t = 0
    max_steps = steps
    max_steps_cap = steps + EXTRA_STEPS_CAP

    while t < max_steps:
        x_in = build_input_vec(img_small_bin, cx, cy, r)
        out  = nn.forward(x_in)
        sx, sy, move_step_nn, sr, rad_step_nn, _ = decode_actions(out, r)

        interior_black = any_black_interior(img255, cx, cy, r)
        border_dir = border_black_direction(img255, cx, cy, r, cos_tab, sin_tab)
        border_has_black = (border_dir is not None)
        all_white = (not interior_black) and (not border_has_black)

        is_complete, inner_b, outer_b = (False, 0.0, 0.0)
        if interior_black:
            is_complete, inner_b, outer_b = circle_complete(img255, cx, cy, r, cos_tab, sin_tab)

        # --- movimento do centro ---
        if all_white:
            if sx == 0 and sy == 0:
                sx, sy = scan_dirs[scan_k % 4]; scan_k += 1
            step = max(1, int(r))
            raw_dx = step * sx; raw_dy = step * sy
            mode = "white_scan"
        elif (not interior_black) and border_has_black:
            step = max(1, int(r))
            ux, uy = border_dir
            raw_dx = step * ux; raw_dy = step * uy
            mode = "border_seek"
        else:
            # aproximar do centróide
            cen = centroid_black_interior(img255, cx, cy, r)
            if cen is not None:
                tx, ty = cen
                dx_f = float(tx - cx); dy_f = float(ty - cy)
                dist = math.hypot(dx_f, dy_f)
                step = int(max(1, min(r, int(round(dist)))))
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    raw_dx = step * ux; raw_dy = step * uy
                else:
                    raw_dx = 0.0; raw_dy = 0.0
            else:
                raw_dx = 0.0; raw_dy = 0.0
            mode = "approach"

        # momentum + clamp centro
        if abs(raw_dx) <= DEADZONE_PX: raw_dx = 0.0
        if abs(raw_dy) <= DEADZONE_PX: raw_dy = 0.0
        vx = MOMENTUM_BETA * vx + (1.0 - MOMENTUM_BETA) * raw_dx
        vy = MOMENTUM_BETA * vy + (1.0 - MOMENTUM_BETA) * raw_dy
        dx = int(round(vx)); dy = int(round(vy))
        cx = cx + dx; cy = cy + dy
        cx, cy = clamp_center_with_radius(size, cx, cy, r)

        # --- ajuste de raio decidido pela NN nos cenários corretos ---
        if interior_black and (not border_has_black):
            # alvo contido — encolher se “pequeno”
            fill = interior_fill_fraction(img255, cx, cy, r)
            if fill < 0.5:
                delta_r = -abs(nn_radius_delta(sr, rad_step_nn))  # força diminuir; magnitude NN
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
                mode = "shrink_nn"
        elif interior_black and border_has_black and (inner_b >= TH_EXPAND_INNER) and (outer_b >= TH_EXPAND_OUTER):
            # dentro e cortando borda — expandir
            delta_r = abs(nn_radius_delta(sr, rad_step_nn))      # força aumentar; magnitude NN
            cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
            mode = "expand_nn"
        elif is_complete:
            # ajuste fino bilateral; limite de magnitude
            raw_dr = nn_radius_delta(sr, rad_step_nn)
            if raw_dr > 0:
                delta_r = min(raw_dr, COMPLETE_REFINE_MAX)
            else:
                delta_r = -min(-raw_dr, COMPLETE_REFINE_MAX)
            if delta_r != 0:
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
                mode = "refine_nn"

        # reforça limites finais
        cx, cy, r = enforce_bounds(size, cx, cy, r)

        # loss + early-stop
        l = metrics_loss_fn(cx, cy, r)
        if (l + IMPROVE_EPS) < best[0]:
            best = (l, cx, cy, r)
            if t >= WARMUP_STEPS: no_improve = 0
        else:
            if t >= WARMUP_STEPS:
                no_improve += 1
                if no_improve >= PATIENCE_STEPS: break

        # steps extras
        if AUTO_EXTEND_STEPS and (max_steps < max_steps_cap):
            if mode in ("approach", "shrink_nn", "expand_nn") and (not is_complete):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode == "border_seek":
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)
            elif mode == "refine_nn":
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_REFINE)

        t += 1

    if return_initial:
        return best[0], best[1], best[2], best[3], initial_state
    return best

# ============================================================
# 11) Controlador com TRACING (GIF) — mesma lógica de raio NN
# ============================================================
def run_controller_trace(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn):
    size = img255.shape[0]
    cx, cy, r = initial_center_fit_all(size)
    cx, cy, r = enforce_bounds(size, cx, cy, r)

    best_loss = metrics_loss_fn(cx, cy, r)
    no_improve = 0

    vx = 0.0; vy = 0.0
    scan_dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    scan_k = 0

    trace = []
    trace.append({"t": -1, "cx": cx, "cy": cy, "r": r, "loss": float(best_loss), "mode": "init"})

    t = 0
    max_steps = steps
    max_steps_cap = steps + EXTRA_STEPS_CAP

    while t < max_steps:
        x_in = build_input_vec(img_small_bin, cx, cy, r)
        out  = nn.forward(x_in)
        sx, sy, move_step_nn, sr, rad_step_nn, _ = decode_actions(out, r)

        interior_black = any_black_interior(img255, cx, cy, r)
        border_dir = border_black_direction(img255, cx, cy, r, cos_tab, sin_tab)
        border_has_black = (border_dir is not None)
        all_white = (not interior_black) and (not border_has_black)

        is_complete, inner_b, outer_b = (False, 0.0, 0.0)
        if interior_black:
            is_complete, inner_b, outer_b = circle_complete(img255, cx, cy, r, cos_tab, sin_tab)

        # movimento do centro
        if all_white:
            if sx == 0 and sy == 0:
                sx, sy = scan_dirs[scan_k % 4]; scan_k += 1
            step = max(1, int(r))
            raw_dx = step * sx; raw_dy = step * sy
            mode = "white_scan"
        elif (not interior_black) and border_has_black:
            step = max(1, int(r))
            ux, uy = border_dir
            raw_dx = step * ux; raw_dy = step * uy
            mode = "border_seek"
        else:
            cen = centroid_black_interior(img255, cx, cy, r)
            if cen is not None:
                tx, ty = cen
                dx_f = float(tx - cx); dy_f = float(ty - cy)
                dist = math.hypot(dx_f, dy_f)
                step = int(max(1, min(r, int(round(dist)))))
                if dist > 0:
                    ux = dx_f / dist; uy = dy_f / dist
                    raw_dx = step * ux; raw_dy = step * uy
                else:
                    raw_dx = 0.0; raw_dy = 0.0
            else:
                raw_dx = 0.0; raw_dy = 0.0
            mode = "approach"

        # momentum + clamp centro
        if abs(raw_dx) <= DEADZONE_PX: raw_dx = 0.0
        if abs(raw_dy) <= DEADZONE_PX: raw_dy = 0.0
        vx = MOMENTUM_BETA * vx + (1.0 - MOMENTUM_BETA) * raw_dx
        vy = MOMENTUM_BETA * vy + (1.0 - MOMENTUM_BETA) * raw_dy
        dx = int(round(vx)); dy = int(round(vy))
        cx = cx + dx; cy = cy + dy
        cx, cy = clamp_center_with_radius(size, cx, cy, r)

        # ajuste de raio pela NN
        if interior_black and (not border_has_black):
            fill = interior_fill_fraction(img255, cx, cy, r)
            if fill < 0.5:
                delta_r = -abs(nn_radius_delta(sr, rad_step_nn))
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
                mode = "shrink_nn"
        elif interior_black and border_has_black and (inner_b >= TH_EXPAND_INNER) and (outer_b >= TH_EXPAND_OUTER):
            delta_r = abs(nn_radius_delta(sr, rad_step_nn))
            cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
            mode = "expand_nn"
        elif is_complete:
            raw_dr = nn_radius_delta(sr, rad_step_nn)
            if raw_dr > 0:
                delta_r = min(raw_dr, COMPLETE_REFINE_MAX)
            else:
                delta_r = -min(-raw_dr, COMPLETE_REFINE_MAX)
            if delta_r != 0:
                cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
                mode = "refine_nn"

        cx, cy, r = enforce_bounds(size, cx, cy, r)

        l = metrics_loss_fn(cx, cy, r)
        trace.append({"t": t, "cx": cx, "cy": cy, "r": r, "loss": float(l), "mode": mode})

        if (l + IMPROVE_EPS) < best_loss:
            best_loss = l
            if t >= WARMUP_STEPS: no_improve = 0
        else:
            if t >= WARMUP_STEPS:
                no_improve += 1
                if no_improve >= PATIENCE_STEPS: break

        # steps extras
        if AUTO_EXTEND_STEPS and (max_steps < max_steps_cap):
            if mode in ("approach", "shrink_nn", "expand_nn") and (not is_complete):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode == "border_seek":
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)
            elif mode == "refine_nn":
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_REFINE)

        t += 1

    best_idx = int(np.argmin([p["loss"] for p in trace]))
    best = trace[best_idx]
    return trace, best

# ============================================================
# 12) Desenho de frames e GIF
# ============================================================
def _draw_frame(img255, p, gt=None, scale=GIF_SCALE):
    cx, cy, r = p["cx"], p["cy"], p["r"]
    t, loss, mode = p["t"], p["loss"], p.get("mode","")
    base = Image.fromarray(img255).convert("RGB")
    if scale != 1:
        base = base.resize((img255.shape[1]*scale, img255.shape[0]*scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)
    bbox_pred = [
        int((cx - r) * scale), int((cy - r) * scale),
        int((cx + r) * scale), int((cy + r) * scale)
    ]
    draw.ellipse(bbox_pred, outline=(255, 0, 0), width=max(1, scale))
    if gt is not None:
        gx, gy, gr = gt
        bbox_gt = [
            int((gx - gr) * scale), int((gy - gr) * scale),
            int((gx + gr) * scale), int((gy + gr) * scale)
        ]
        draw.ellipse(bbox_gt, outline=(0, 200, 0), width=max(1, scale))
    draw.text((5, 5), f"t={t} loss={loss:.4f} ({cx},{cy},r={r}) mode={mode}", fill=(255,255,0))
    return base

def save_gif_for_trace(img255, trace, gt_tuple, out_path):
    frames = [_draw_frame(img255, p, gt_tuple) for p in trace]
    if len(frames) == 1: frames = frames * 2
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=False,
        disposal=2
    )

# ============================================================
# 13) Config da rede / persistência
# ============================================================
input_size   = IN_SIZE * IN_SIZE + 3
hidden_sizes = [6]
output_size  = 22
nn = NeuralNetwork(input_size, hidden_sizes, output_size)

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
        "version": "2025-10-29.v7_nn_controls_radius_amount_bounded",
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
# 14) Carrega anotações
# ============================================================
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]
annotations = [rec for rec in records if rec.get("split") == "single"]

# ============================================================
# 15) Loop GA + execução final (com GIFs)
# ============================================================
initial_population = None
_num_weights = nn.get_weights().size
_num_bits    = int(_num_weights * 16)

ckpt_weights, ckpt_pop = load_checkpoint(_num_weights, _num_bits, CHECKPOINT_PATH)
if ckpt_weights is not None:
    nn.set_weights(ckpt_weights.astype(np.float32))
    print(f"[ckpt] Pesos carregados: {CHECKPOINT_PATH} (num_weights={_num_weights})")
if ckpt_pop is not None:
    initial_population = ckpt_pop.astype(np.uint8)
    print(f"[ckpt] População GA carregada: {initial_population.shape}")

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

    # ---------- Stage 0: loss inicial ----------
    base_loss, cx0, cy0, r0, initial_state = run_controller(
        nn, img_full, img_small,
        steps=0,
        cos_tab=COS_COARSE, sin_tab=SIN_COARSE,
        metrics_loss_fn=metrics_loss_coarse,
        probe_r_list=probe_list_coarse,
        return_initial=True
    )
    cx_init, cy_init, r_init, loss_init_true = initial_state

    # ---------- Stage 1: GA coarse ----------
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

    # ---------- Stage 2: GA polish (fine) ----------
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

    # ---------- Stage 3: Execução final (fine) + TRACING/GIF ----------
    trace, best = run_controller_trace(
        nn, img_full, img_small,
        steps=steps_fine,
        cos_tab=COS_FINE, sin_tab=SIN_FINE,
        metrics_loss_fn=metrics_loss_fine
    )
    final_loss = float(best["loss"])
    cx_pred, cy_pred, r_pred = int(best["cx"]), int(best["cy"]), int(best["r"])

    fill     = interior_fill_fraction(img_full, cx_pred, cy_pred, r_pred)
    inner_b  = _ring_fraction_vec(img_full, cx_pred, cy_pred, r_pred, delta=-1, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    outer_b  = _ring_fraction_vec(img_full, cx_pred, cy_pred, r_pred, delta=+1, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    cut      = _border_cut_vec(img_full, cx_pred, cy_pred, r_pred, cos_tab=COS_FINE, sin_tab=SIN_FINE)
    probe_b  = _probe_max_thick(img_full, cx_pred, cy_pred, _make_probe_list(PROBE_R_FINE, IMG_SIZE),
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

    print(f"Imagem: {file_path}")
    print(f"GT (px):        (x={x_real}, y={y_real}, r={r_real})")
    print(f"Inicial (px):   (x={cx_init}, y={cy_init}, r={r_init}), loss_inicial={loss_init_true:.6f}")
    print(f"Predição (px):  (x={cx_pred}, y={cy_pred}, r={r_pred})")
    print(f"Loss final:     {final_loss:.6f} | fill={fill:.4f} inner={inner_b:.4f} outer={outer_b:.4f} cut={cut:.4f} probe={probe_b:.4f} IoU={iou:.4f}")
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
                 "policy": "white_scan | border_seek | approach | shrink_nn | expand_nn | refine_nn",
                 "auto_extend": AUTO_EXTEND_STEPS},
        "gif": gif_path, "time": time.time()
    }
    logf.write(json.dumps(log_line, ensure_ascii=False) + "\n")
    logf.flush()

# resumo do dataset
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
            "POLICY": "white_scan | border_seek | approach | shrink_nn | expand_nn | refine_nn",
            "TH_INNER": TH_INNER, "TH_OUTER": TH_OUTER,
            "TH_EXPAND_INNER": TH_EXPAND_INNER, "TH_EXPAND_OUTER": TH_EXPAND_OUTER,
            "RAD_STEP_FLOOR_PX": RAD_STEP_FLOOR_PX, "COMPLETE_REFINE_MAX": COMPLETE_REFINE_MAX,
            "AUTO_EXTEND_STEPS": AUTO_EXTEND_STEPS,
            "EXTRA_STEPS_HAS_SIGNAL": EXTRA_STEPS_HAS_SIGNAL,
            "EXTRA_STEPS_BORDER": EXTRA_STEPS_BORDER,
            "EXTRA_STEPS_REFINE": EXTRA_STEPS_REFINE,
            "EXTRA_STEPS_CAP": EXTRA_STEPS_CAP
        }
    }
    with open(RUN_JSONL_PATH, 'a', encoding='utf-8') as fsum:
        fsum.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(f"[Logs] JSONL salvo em: {RUN_JSONL_PATH}")
