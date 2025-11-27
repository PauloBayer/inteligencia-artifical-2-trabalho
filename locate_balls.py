#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação (somente inferência) do detector GA+RNA de bolas pretas:
- Carrega pesos de um checkpoint (.npz) treinado pelo index.py
- Roda o controlador para cada imagem do dataset
- Faz busca sequencial de múltiplas bolas por imagem (mascarando as já encontradas)
- Gera UM GIF por imagem mostrando toda a caçada, incluindo tentativas MISS
"""

import os
import json
import time
import math
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 0) Configuração global
# ============================================================

parser = argparse.ArgumentParser(
    description="Avaliação somente inferência do detector GA+RNA de bolas pretas."
)

parser.add_argument(
    "--data-root",
    type=str,
    default="amostras/dados",
    help="Diretório raiz do dataset (onde está annotations.jsonl).",
)

parser.add_argument(
    "--split",
    choices=["single", "multi", "both"],
    default="multi",
    help="Qual split do dataset usar: single, multi ou both.",
)

parser.add_argument(
    "--ckpt",
    type=str,
    default="checkpoints/ckpt_single_ga_controller_v13_fast_smallnet_iouchase_borderaware_initpolicy_random_then_nn_inithead_0.npz",
    help="Caminho para o checkpoint .npz treinado pelo index.py.",
)

parser.add_argument(
    "--make-gifs",
    action="store_true",
    default=True,
    help="Se passado, gera GIFs (um por imagem).",
)

parser.add_argument(
    "--no-gifs",
    action="store_true",
    help="Se passado, desativa GIFs, mesmo que --make-gifs seja usado.",
)

parser.add_argument(
    "--gif-limit",
    type=int,
    default=None,
    help="Máximo de GIFs para gerar (None = sem limite).",
)

parser.add_argument(
    "--infer-mult",
    type=int,
    default=5,
    help="Fator multiplicativo dos passos de controle na fase de inferência.",
)

cli_args = parser.parse_args()

rng = np.random.default_rng()

IN_SIZE  = 28
IMG_SIZE = 255
R_NORM   = int(math.ceil(math.hypot(IMG_SIZE - 1, IMG_SIZE - 1)))

DATA_ROOT        = cli_args.data_root
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, "annotations.jsonl")
SPLIT_CHOICE     = cli_args.split
CKPT_PATH        = cli_args.ckpt

# Binarização simples no full-res
BLACK_THR = 64

# Parada por acerto perfeito
STOP_ON_PERFECT      = True
PERFECT_INNER_FRAC   = 0.995
PERFECT_OUTER_FRAC   = 0.005
PERFECT_THICKNESS    = 1

# Parada por IoU com máscara preta
MASK_IOU_STOP_ENABLE = True
MASK_IOU_STOP_THR    = 0.94

# Config GIF / viz
SHOW_GT_IN_GIF       = True
PRED_ON_TOP          = True

TRIM_GIF_AT_STOP     = True
KEEP_SNAP_AFTER_STOP = True

HUD_TEXT_COLOR   = (255, 255, 255)
HUD_STROKE_COLOR = (0, 0, 0)
HUD_BG_RGBA      = None
HUD_FONT_PATH    = None
HUD_FONT_SIZE    = 10
HUD_PAD          = 2
HUD_POS          = (5, 5)

# Estratégias no branco
WHITE_STREAK_GROWTH   = 0.80
WHITE_JITTER_PX       = 3
WHITE_JITTER_CLAMP    = 24
WHITE_SUPERJUMP_EVERY = 10
WHITE_SUPERJUMP_PX    = IMG_SIZE // 2

WHITE_RADIUS_GROW_AFTER    = 20
WHITE_RADIUS_GROW_EVERY    = 4
WHITE_RADIUS_GROW_MAX_PX   = 6
WHITE_RADIUS_REQUIRE_BIT   = True
WHITE_RADIUS_RESET_STREAK  = False

APPROACH_STEP_GAIN_FAR     = 0.60
APPROACH_STEP_GAIN_NEAR    = 0.35
APPROACH_NEAR_FRAC_OF_R    = 0.75
APPROACH_STEP_MAX_PX       = 32
BORDER_STEP_MAX_PX         = 24
MOMENTUM_DAMP_ON_MODE_CHANGE = 0.25

EPS_OUTER_EXPAND = 0.02
EPS_INNER_SHRINK = 0.4

RAD_CAP_NEAR_PX = 1
RAD_CAP_FAR_PX  = 8

IOU_EVAL_STOP      = False
IOU_EVAL_THRESH    = 1.0
IOU_CHASE_ENABLE   = True
IOU_IMPROVE_DELTA  = 0.001
IOU_STEPS_BONUS    = 64
IOU_EXTRA_CAP      = 2000

PINGPONG_TOL_PX         = 1
PINGPONG_REQUIRE_MODE   = True
PINGPONG_ONLY_WHEN_NEAR = True

# ============================================================
# 1) Hiperparâmetros de controle
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
TH_EXPAND_INNER     = 0.85
TH_EXPAND_OUTER     = 0.50

AUTO_EXTEND_STEPS       = True
EXTRA_STEPS_HAS_SIGNAL  = 8
EXTRA_STEPS_BORDER      = 6
EXTRA_STEPS_REFINE      = 4
EXTRA_STEPS_CAP         = 24
EXTRA_STEPS_CAP_TRACE   = 2000

SHRINK_GAIN_MIN      = 1.25
SHRINK_GAIN_MAX      = 4.0
SHRINK_CAP_NEAR_PX   = 8
SHRINK_CAP_FAR_PX    = 14

ALLOW_PARTIAL_CIRCLE = True
R_EXT_MAX = 40

EARLY_STOP = False
SEARCH_BUDGET = "balanced"

INFER_STEPS_MULT = int(cli_args.infer_mult)

# ============================================================
# 1.2) Inicialização / movimento
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

# Infra de logs/resultados e GIFs
RUNS_DIR          = "runs_eval"
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_ID            = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_JSONL_PATH    = os.path.join(RUNS_DIR, f"eval_{RUN_ID}.jsonl")

MAKE_GIFS         = bool(cli_args.make_gifs and not cli_args.no_gifs)
GIFS_DIR          = os.path.join(RUNS_DIR, "gifs")
if MAKE_GIFS:
    os.makedirs(GIFS_DIR, exist_ok=True)
GIF_SCALE         = 2
GIF_DURATION_MS   = 120
GIF_TAIL_HOLD_MS  = 5000
GIF_TAIL_COMPAT_DUPLICATE  = False
GIF_TAIL_DUPLICATE_EACH_MS = 250
GIF_LIMIT         = cli_args.gif_limit

# ============================================================
# 2) I/O de imagem
# ============================================================
def load_image_small_bin(file_path, out_size=IN_SIZE):
    img = Image.open(file_path).convert('L').resize((out_size, out_size), Image.NEAREST)
    arr = np.array(img)
    return np.where(arr < 128, 1.0, 0.0).astype(np.float32)

def load_image_full_gray(file_path):
    return np.array(Image.open(file_path).convert('L'))

# ============================================================
# 3) Anéis e métricas
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
    xs = np.rint(cx + rr * cos_tab).astype(np.int32)
    ys = np.rint(cy + rr * sin_tab).astype(np.int32)
    total = xs.size
    if total == 0:
        return 0.0
    valid = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    if not np.any(valid):
        black = 0
    else:
        vals = img255[ys[valid], xs[valid]]
        black = int(np.count_nonzero(vals <= BLACK_THR))
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
# 3.b) IoU com máscara preta
# ============================================================
def black_mask(img255, thr=BLACK_THR):
    return (img255 <= thr)

def iou_circle_vs_mask(img255, cx, cy, r, thr=BLACK_THR):
    size = img255.shape[0]
    cm = _circle_mask(size, int(cx), int(cy), int(max(1, r)))
    bm = black_mask(img255, thr)
    inter = int(np.count_nonzero(cm & bm))
    union = int(np.count_nonzero(cm | bm))
    return (inter / union) if union > 0 else 0.0

def r_fit_for_center(size, cx, cy):
    return int(max(0, min(cx, cy, size-1-cx, size-1-cy)))

def enforce_bounds_partial(size, cx, cy, r):
    cx = int(np.clip(cx, 0, size - 1))
    cy = int(np.clip(cy, 0, size - 1))
    r = int(max(1, min(r, R_EXT_MAX)))
    return cx, cy, r

def snap_refine_mask_iou(img255, cx, cy, r, *,
                         dxy=1, dr=2, thr=BLACK_THR,
                         prefer_smaller_radius=True):
    size = img255.shape[0]
    best_score = iou_circle_vs_mask(img255, cx, cy, r, thr)
    best = (int(cx), int(cy), int(r))
    for dy in range(-dxy, dxy + 1):
        for dx in range(-dxy, dxy + 1):
            cx2 = int(np.clip(cx + dx, 0, size - 1))
            cy2 = int(np.clip(cy + dy, 0, size - 1))
            rmax2 = r_fit_for_center(size, cx2, cy2)
            for dr_ in range(-dr, dr + 1):
                r2 = int(np.clip(r + dr_, 1, max(1, rmax2)))
                s = iou_circle_vs_mask(img255, cx2, cy2, r2, thr)
                if (s > best_score) or (abs(s - best_score) < 1e-12 and prefer_smaller_radius and r2 < best[2]):
                    best_score = s
                    best = (cx2, cy2, r2)
    return best, best_score

# ============================================================
# 3.c) Máscaras e checagens para busca sequencial
# ============================================================
def array_to_image_small_bin(img_array_full, out_size=IN_SIZE):
    img_pil = Image.fromarray(img_array_full.astype(np.uint8)) 
    img_small = img_pil.resize((out_size, out_size), Image.NEAREST).convert('L')
    arr = np.array(img_small)
    return np.where(arr < 128, 1.0, 0.0).astype(np.float32)

def mask_circle_in_image(img255, cx, cy, r, val=255):
    size = img255.shape[0]
    mask = _circle_mask(size, cx, cy, r)
    img255[mask] = val
    return img255

def check_for_black(img255, thr=BLACK_THR):
    return np.any(img255 <= thr)

# ============================================================
# 4) Probes + loss
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
# 5) MLP
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
# 6) Decodificação de ações
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
# 7) Estado / inicialização / limites
# ============================================================
def build_input_vec(img_small_bin, cx, cy, r):
    state = np.array([cx/IMG_SIZE, cy/IMG_SIZE, r/float(R_NORM)], dtype=np.float32)
    return np.concatenate([img_small_bin.flatten(), state], axis=0).astype(np.float32)

def initial_center_fit_all(size):
    cx = size // 2; cy = size // 2
    r  = r_fit_for_center(size, cx, cy)
    return cx, cy, r

def clamp_center_partial(size, cx, cy):
    cx = int(np.clip(cx, 0, size - 1))
    cy = int(np.clip(cy, 0, size - 1))
    return cx, cy

def initial_center_random(size, rng, r_min_px=INIT_RANDOM_R_MIN_PX, r_max_frac=INIT_RANDOM_R_MAX_FRAC):
    r_max_px = int(max(1, min(int(r_max_frac * size), size // 2)))
    r = int(rng.integers(low=max(1, r_min_px), high=r_max_px + 1))
    cx = int(rng.integers(low=r, high=size - r))
    cy = int(rng.integers(low=r, high=size - r))
    return cx, cy, r

def nn_radius_delta(sr, rad_step):
    step = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step))))
    return int(sr) * step

def apply_radius_partial(size, cx, cy, r, delta_r):
    r_new = max(1, r + int(delta_r))
    return enforce_bounds_partial(size, cx, cy, r_new)

def apply_radius_recenter_partial(size, cx, cy, r, delta_r):
    return apply_radius_partial(size, cx, cy, r, delta_r)

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

    if ALLOW_PARTIAL_CIRCLE:
        cx, cy = clamp_center_partial(size, cx, cy)
        delta_r = nn_radius_delta(sr, rad_step_nn)
        cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
        cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
    else:
        cx, cy, r = initial_center_fit_all(size)

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
    cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
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

    if ALLOW_PARTIAL_CIRCLE:
        return enforce_bounds_partial(size, cx, cy, r)
    else:
        return initial_center_fit_all(size)

# ============================================================
# 8) Heurísticas auxiliares
# ============================================================
def compute_shrink_delta(rad_step_nn, inner_b, near):
    base = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
    deficit = 0.0
    if EPS_INNER_SHRINK > 1e-9:
        deficit = max(0.0, (EPS_INNER_SHRINK - inner_b) / float(EPS_INNER_SHRINK))
        deficit = min(deficit, 1.0)
    gain = SHRINK_GAIN_MIN + (SHRINK_GAIN_MAX - SHRINK_GAIN_MIN) * deficit
    cap = SHRINK_CAP_NEAR_PX if near else SHRINK_CAP_FAR_PX
    mag = min(int(round(gain * base)), cap)
    return -mag

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
# 9) Controlador (sem traço)
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

        if ALLOW_PARTIAL_CIRCLE:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
        else:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)

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
                    if ALLOW_PARTIAL_CIRCLE:
                        new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
                    else:
                        new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        white_growth_done = False
        if all_white:
            grow_ready = (white_streak >= WHITE_RADIUS_GROW_AFTER)
            if grow_ready:
                periodic_ok = ((white_streak - WHITE_RADIUS_GROW_AFTER) % max(1, WHITE_RADIUS_GROW_EVERY) == 0)
                bit_ok = True
                if periodic_ok and bit_ok:
                    rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                    grow_px = int(min(rad_mag, WHITE_RADIUS_GROW_MAX_PX))
                    if grow_px > 0:
                        if ALLOW_PARTIAL_CIRCLE:
                            cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
                        else:
                            cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
                        if WHITE_RADIUS_RESET_STREAK:
                            white_streak = 0
                        white_growth_done = True

        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)

        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if interior_black and (inner_b < EPS_INNER_SHRINK):
                delta_r = compute_shrink_delta(rad_step_nn, inner_b, near)
                if delta_r < 0:
                    delta_r = -min(abs(delta_r), cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
            elif outer_b > EPS_OUTER_EXPAND:
                rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                delta_r = +min(rad_mag, cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)

        if ALLOW_PARTIAL_CIRCLE:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
        else:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)

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
            if (outer_b > EPS_OUTER_EXPAND) or (interior_black and inner_b < EPS_INNER_SHRINK) or (mode in ("approach",)):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode in ("border_seek",):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)

        prev_mode = mode
        t += 1

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
# 10) Controlador com trace (para GIF)
# ============================================================
def run_controller_trace(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, gt_tuple=None,
                         cx_start=None, cy_start=None, r_start=None):
    size = img255.shape[0]
    
    if cx_start is not None and cy_start is not None and r_start is not None:
        cx, cy, r = cx_start, cy_start, r_start
        if ALLOW_PARTIAL_CIRCLE:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
        else:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
    else:
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

        if ALLOW_PARTIAL_CIRCLE:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
        else:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)

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
                    if ALLOW_PARTIAL_CIRCLE:
                        new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
                    else:
                        new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        white_growth_done = False
        mode_r = None

        if all_white:
            grow_ready = (white_streak >= WHITE_RADIUS_GROW_AFTER)
            if grow_ready:
                periodic_ok = ((white_streak - WHITE_RADIUS_GROW_AFTER) % max(1, WHITE_RADIUS_GROW_EVERY) == 0)
                if periodic_ok:
                    rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                    grow_px = int(min(rad_mag, WHITE_RADIUS_GROW_MAX_PX))
                    if grow_px > 0:
                        if ALLOW_PARTIAL_CIRCLE:
                            cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
                        else:
                            cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
                        white_growth_done = True
                        mode_r = "white_grow"

        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)

        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if interior_black and (inner_b < EPS_INNER_SHRINK):
                delta_r = compute_shrink_delta(rad_step_nn, inner_b, near)
                if delta_r < 0:
                    delta_r = -min(abs(delta_r), cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
                mode_r = "shrink_nn"
            elif outer_b > EPS_OUTER_EXPAND:
                rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                delta_r = +min(rad_mag, cap_px) 
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)
                mode_r = "expand_nn"

        if ALLOW_PARTIAL_CIRCLE:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
        else:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)

        l = metrics_loss_fn(cx, cy, r)

        current_iou = 0.0
        if MASK_IOU_STOP_ENABLE:
            current_iou = iou_circle_vs_mask(img255, cx, cy, r, thr=BLACK_THR)
        if MASK_IOU_STOP_ENABLE and current_iou >= MASK_IOU_STOP_THR:
            mode_final = "mask_iou_stop"
            trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                          "loss": float(l), "mode": mode_final})
            if (l + IMPROVE_EPS) < best_loss: best_loss = l
            break

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

    (snap_cx, snap_cy, snap_r), _ = snap_refine_mask_iou(
        img255, best["cx"], best["cy"], best["r"],
        dxy=1, dr=8, thr=BLACK_THR, prefer_smaller_radius=True
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
# 11) Desenho e GIF único por imagem
# ============================================================
def _draw_frame(img255, p, gt=None, scale=GIF_SCALE):
    cx, cy, r = p["cx"], p["cy"], p["r"]
    t, loss, mode = p["t"], p["loss"], p.get("mode","")

    base = Image.fromarray(img255).convert("RGB")
    if scale != 1:
        base = base.resize((img255.shape[1]*scale, img255.shape[0]*scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)

    bbox_pred = [int((cx - r) * scale), int((cy - r) * scale),
                 int((cx + r) * scale), int((cy + r) * scale)]

    bbox_gt = None
    if gt is not None:
        gx, gy, gr = gt
        bbox_gt = [int((gx - gr) * scale), int((gy - gr) * scale),
                   int((gx + gr) * scale), int((gy + gr) * scale)]

    w_pred = max(1, scale)
    w_gt   = max(1, w_pred - 1)

    RED   = (255, 0, 0)
    GREEN = (0, 200, 0)

    if PRED_ON_TOP:
        if SHOW_GT_IN_GIF and (bbox_gt is not None):
            draw.ellipse(bbox_gt, outline=GREEN, width=w_gt)
        draw.ellipse(bbox_pred, outline=RED, width=w_pred)
    else:
        draw.ellipse(bbox_pred, outline=RED, width=w_pred)
        if SHOW_GT_IN_GIF and (bbox_gt is not None):
            draw.ellipse(bbox_gt, outline=GREEN, width=w_gt)

    font = None
    try:
        if HUD_FONT_PATH:
            font = ImageFont.truetype(HUD_FONT_PATH, size=max(8, int(HUD_FONT_SIZE * scale)))
    except Exception:
        font = None
    if font is None:
        font = ImageFont.load_default()

    loss_txt = f"{loss:.4f}" if isinstance(loss, (float,int)) and not math.isnan(loss) else "NA"
    hud_text = f"t={t} loss={loss_txt} ({cx},{cy},r={r}) mode={mode}"

    x0 = int(HUD_POS[0] * scale)
    y0 = int(HUD_POS[1] * scale)
    sw = max(1, scale // 2)

    if HUD_BG_RGBA is not None:
        tmp_draw = ImageDraw.Draw(base)
        left, top, right, bottom = tmp_draw.textbbox((x0, y0), hud_text, font=font, stroke_width=sw)
        pad = int(HUD_PAD * scale)
        rect = (left - pad, top - pad, right + pad, bottom + pad)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle(rect, fill=HUD_BG_RGBA)
        odraw.text((x0, y0), hud_text,
                   font=font,
                   fill=HUD_TEXT_COLOR,
                   stroke_width=sw,
                   stroke_fill=HUD_STROKE_COLOR)
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    else:
        draw.text((x0, y0), hud_text,
                  font=font,
                  fill=HUD_TEXT_COLOR,
                  stroke_width=sw,
                  stroke_fill=HUD_STROKE_COLOR)

    return base

STOP_MODES_ORDERED = ["iou_stop_equal", "iou_stop", "mask_iou_stop", "radius_ping_pong", "perfect_stop"]

def _first_stop_index(trace):
    best = None
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
    idx, _ = _first_stop_index(trace)
    if idx is None:
        return trace
    end = idx
    if keep_snap and (idx + 1) < len(trace):
        nxt = trace[idx + 1]
        if nxt.get("mode") == "snap_refine" and nxt.get("t") == trace[idx].get("t"):
            end = idx + 1
    return trace[:end + 1]

def save_gif_for_frames(frames_info, gt_tuple, out_path):
    """
    Gera um GIF único por imagem a partir de vários frames:
      frames_info: lista de (img255_snapshot, p_dict)

    A seleção de quais tentativas entram (apenas acertos ou todas as falhas)
    é feita fora desta função; aqui só desenhamos e salvamos.
    """
    if not frames_info:
        return

    gt_draw = gt_tuple if SHOW_GT_IN_GIF else None

    frames = []
    for img255, p in frames_info:
        frames.append(_draw_frame(img255, p, gt_draw))

    if len(frames) == 1:
        # GIF com 1 frame pode bugar em alguns viewers -> duplica
        frames = frames * 2

    durations = [int(GIF_DURATION_MS)] * len(frames)

    if GIF_TAIL_COMPAT_DUPLICATE:
        hold_ms = max(0, int(GIF_TAIL_HOLD_MS))
        if hold_ms > 0:
            last = frames[-1]
            frames = frames[:-1]
            durations = durations[:-1]
            n = max(1, int(math.ceil(hold_ms / float(GIF_TAIL_DUPLICATE_EACH_MS))))
            frames.extend([last] * n)
            durations.extend([int(GIF_TAIL_DUPLICATE_EACH_MS)] * n)
    else:
        if len(durations) > 0:
            durations[-1] = int(GIF_TAIL_HOLD_MS)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2
    )

# ============================================================
# 12) Configuração da rede / load checkpoint
# ============================================================
input_size   = IN_SIZE * IN_SIZE + 3
hidden_sizes = [4]
output_size  = ACTION_BITS + (3 if USE_INIT_HEAD else 0)
nn = NeuralNetwork(input_size, hidden_sizes, output_size, use_init_head=USE_INIT_HEAD)

def load_checkpoint_weights(path, expected_num_weights):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    data = np.load(path, allow_pickle=False)
    weights = data.get("weights", None)
    if weights is None:
        raise RuntimeError(f"Checkpoint {path} não tem array 'weights'.")
    if weights.size != expected_num_weights:
        raise RuntimeError(f"Checkpoint {path} tem {weights.size} pesos, esperado {expected_num_weights}.")
    return weights.astype(np.float32)

_num_weights = nn.get_weights().size
ckpt_weights = load_checkpoint_weights(CKPT_PATH, _num_weights)
nn.set_weights(ckpt_weights)
print(f"[ckpt] Weights carregados de {CKPT_PATH} (num_weights={_num_weights})")

# ============================================================
# 13) Carrega anotações
# ============================================================
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

if SPLIT_CHOICE == "both":
    splits_ativos = ("single", "multi")
else:
    splits_ativos = (SPLIT_CHOICE,)

annotations = [rec for rec in records if rec.get("split") in splits_ativos]

print(f"[dataset] DATA_ROOT={DATA_ROOT} | split={SPLIT_CHOICE} | imagens={len(annotations)}")

# ============================================================
# 14) Loop de avaliação + busca sequencial + GIF único
# ============================================================
logf = open(RUN_JSONL_PATH, 'w', encoding='utf-8')
gif_count = 0

STOP_LOSS_THR = 1.5
MIN_BLACK_PIXELS = 10
MIN_ACCEPTED_FILL = 0.85
MAX_BAD_CANDIDATES_IN_A_ROW = 5

sum_iou = 0.0
cnt = 0
cnt_iou_good = 0
cnt_stuck = 0  # mantido por compat

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
    fine_weights   = (5.0, 1.5, 1.0, 5.0)

    metrics_loss_coarse = make_metrics_loss(
        img_full, COS_COARSE, SIN_COARSE, cache_coarse, probe_list_coarse,
        weights=coarse_weights, w_probe=W_PROBE_COARSE
    )
    metrics_loss_fine   = make_metrics_loss(
        img_full, COS_FINE,   SIN_FINE,   cache_fine,   probe_list_fine,
        weights=fine_weights, w_probe=W_PROBE_FINE
    )

    # Referência: usamos a primeira bola do GT só para ter um alvo "referência" em logs/GIF
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

    # Ajuste simples de steps_fine com base na qualidade inicial
    if loss_init_true <= 0.6:
        steps_fine = max(10, CTRL_STEPS_FINE - 2)
    elif loss_init_true <= 1.2:
        steps_fine = CTRL_STEPS_FINE
    else:
        steps_fine = max(CTRL_STEPS_FINE, 16)

    # ============================================================
    # Busca sequencial em current_img_full
    # REGRA DO GIF:
    #   - Só tentativas ACEITAS entram em frames_info.
    #   - Se nenhuma bola for aceita, o GIF mostra TODAS as tentativas falhas.
    #   - Se houver bolas aceitas, adicionamos também a ÚLTIMA tentativa falha.
    # ============================================================
    current_img_full = img_full.copy()
    balls_found = []

    # frames das tentativas aceitas (mais erase)
    frames_info = []

    # frames de TODAS as tentativas falhas (usadas só se nenhuma bola for aceita)
    failed_attempts_frames = []

    # frames da ÚLTIMA tentativa falha (para anexar depois da última bola válida)
    last_failed_attempt_frames = None

    has_any_valid_ball = False
    bad_candidates_in_a_row = 0

    print(f"\n--- Iniciando busca sequencial para {file_rel} ---")

    while check_for_black(current_img_full):
        num_black = np.count_nonzero(current_img_full <= BLACK_THR)
        if num_black < MIN_BLACK_PIXELS:
            print(f"  [STOP] Pixels pretos insuficientes restantes ({num_black}).")
            break

        current_img_small = array_to_image_small_bin(current_img_full, out_size=IN_SIZE)

        size = current_img_full.shape[0]
        ys, xs = np.where(current_img_full <= BLACK_THR)
        if xs.size == 0:
            print("  [STOP] Nenhum pixel preto encontrado (xs.size == 0).")
            break

        # centro inicial: média dos pixels pretos restantes
        cx_start = int(np.clip(int(np.rint(xs.mean())), 0, size - 1))
        cy_start = int(np.clip(int(np.rint(ys.mean())), 0, size - 1))
        r_start  = max(1, INIT_RANDOM_R_MIN_PX)

        cache_seq = {}
        metrics_loss_seq = make_metrics_loss(
            current_img_full,
            COS_FINE, SIN_FINE,
            cache_seq,
            probe_list_fine,
            weights=fine_weights,
            w_probe=W_PROBE_FINE
        )

        # snapshot da imagem ANTES de apagar bola (para os frames desta tentativa)
        img_for_gif = current_img_full.copy()

        trace, best = run_controller_trace(
            nn, current_img_full, current_img_small,
            steps=int(steps_fine * max(1, INFER_STEPS_MULT)),
            cos_tab=COS_FINE, sin_tab=SIN_FINE,
            metrics_loss_fn=metrics_loss_seq,
            gt_tuple=(x_real, y_real, r_real),
            cx_start=cx_start, cy_start=cy_start, r_start=r_start 
        )

        # Para o GIF: usamos o trace aparado no ponto de parada (quando houver)
        trace_for_gif = trim_trace_on_stop(trace, keep_snap=KEEP_SNAP_AFTER_STOP)
        attempt_frames = [(img_for_gif.copy(), p) for p in trace_for_gif]

        final_loss = float(best["loss"])
        cx_pred, cy_pred, r_pred = int(best["cx"]), int(best["cy"]), int(best["r"])

        final_fill = interior_fill_fraction(current_img_full, cx_pred, cy_pred, r_pred)
        is_valid_ball = (final_loss <= STOP_LOSS_THR) and (final_fill >= MIN_ACCEPTED_FILL)

        # CASO 1: CANDIDATA RUIM
        #   - Não apaga nada.
        #   - Frames vão para failed_attempts_frames.
        #   - last_failed_attempt_frames guarda SEMPRE a última tentativa falha.
        if not is_valid_ball:
            bad_candidates_in_a_row += 1
            print(
                f"  [MISS] Candidata ruim #{bad_candidates_in_a_row} "
                f"(loss={final_loss:.4f}, fill={final_fill:.4f})."
            )

            # guarda esta tentativa entre as falhas (para caso sem bolas válidas)
            failed_attempts_frames.extend(attempt_frames)

            # e marca esta como a ÚLTIMA tentativa falha
            last_failed_attempt_frames = attempt_frames

            if bad_candidates_in_a_row >= MAX_BAD_CANDIDATES_IN_A_ROW:
                print(
                    f"  [STOP] {bad_candidates_in_a_row} candidatas ruins seguidas. "
                    f"Encerrando busca nesta imagem."
                )
                break

            continue

        # CASO 2: BOLA ACEITA
        #   - Zera contador de ruins.
        #   - Marca que há pelo menos uma bola válida.
        #   - Frames desta tentativa vão para frames_info.
        #   - Apaga bola encontrada e adiciona frame "erase".
        bad_candidates_in_a_row = 0
        has_any_valid_ball = True

        balls_found.append({"x": cx_pred, "y": cy_pred, "r": r_pred, "loss": final_loss})

        if MAKE_GIFS and (GIF_LIMIT is None or gif_count < GIF_LIMIT):
            # frames da tentativa aceita (antes de apagar)
            frames_info.extend(attempt_frames)

        # apaga a bola para procurar a próxima
        current_img_full = mask_circle_in_image(current_img_full, cx_pred, cy_pred, r_pred)

        if MAKE_GIFS and (GIF_LIMIT is None or gif_count < GIF_LIMIT):
            # frame extra mostrando a imagem após o erase
            erase_state = dict(best)
            erase_state["t"] = erase_state.get("t", 0) + 1
            erase_state["mode"] = str(erase_state.get("mode", "")) + "|erase"
            frames_info.append((current_img_full.copy(), erase_state))

    # ============================================================
    # GIF único por imagem:
    #   - Se houve pelo menos uma bola aceita:
    #         frames_info (todas aceitas + erase) + última tentativa falha (se houver).
    #   - Se nenhuma bola foi aceita:
    #         todas as tentativas falhas (failed_attempts_frames).
    # ============================================================
    if has_any_valid_ball:
        frames_to_use = list(frames_info)  # cópia
        if last_failed_attempt_frames is not None:
            frames_to_use.extend(last_failed_attempt_frames)
    else:
        frames_to_use = failed_attempts_frames

    if MAKE_GIFS and (GIF_LIMIT is None or gif_count < GIF_LIMIT) and len(frames_to_use) > 0:
        safe_name = file_rel.replace("/", "__")
        gif_path = os.path.join(
            GIFS_DIR,
            f"{os.path.splitext(safe_name)[0]}_{RUN_ID}.gif"
        )
        save_gif_for_frames(frames_to_use, (x_real, y_real, r_real), gif_path)
        gif_count += 1

        if has_any_valid_ball:
            print(f"  [gif] GIF único por imagem salvo em: {gif_path} (bolas válidas + ÚLTIMA tentativa falha)")
        else:
            print(f"  [gif] GIF único por imagem salvo em: {gif_path} (somente tentativas FALHAS; nenhuma bola aceita)")

    # ============================================================
    # Métricas e log
    # ============================================================
    cnt += 1

    print(f"\nResultado Final para Imagem: {file_path}")
    print(f"GT (px) ref:    (x={x_real}, y={y_real}, r={r_real})")
    print(f"Bolas Encontradas: {len(balls_found)}")
    
    total_iou_img = 0.0
    
    for i, ball in enumerate(balls_found):
        iou_val = iou_circle(IMG_SIZE, (ball["x"], ball["y"], ball["r"]), (x_real, y_real, r_real))
        total_iou_img += iou_val
        fill = interior_fill_fraction(img_full, ball["x"], ball["y"], ball["r"])
        
        print(f"  -> Bola {i+1} (px): (x={ball['x']}, y={ball['y']}, r={ball['r']})")
        print(f"     Loss: {ball['loss']:.4f} | Fill: {fill:.4f} | IoU vs GT1: {iou_val:.4f}")

    if len(balls_found) > 0:
        sum_iou += (total_iou_img / len(balls_found))
        if total_iou_img > 0.5:
            cnt_iou_good += 1 
    
    log_line = {
        "run_id": RUN_ID,
        "file": file_rel,
        "gt_ref": {"x": x_real, "y": y_real, "r": r_real},
        "init": {"x": cx_init, "y": cy_init, "r": r_init, "loss": float(loss_init_true)},
        "predicoes": balls_found,
        "total_bolas_encontradas": len(balls_found),
        "ctrl": {"coarse_steps": CTRL_STEPS_COARSE, "fine_steps": steps_fine,
                 "policy": "SEQUENCIAL_SEARCH_EVAL_ONLY",
                 "search_budget": SEARCH_BUDGET, "infer_steps_mult": INFER_STEPS_MULT},
        "time": time.time()
    }
    logf.write(json.dumps(log_line, ensure_ascii=False) + "\n")
    logf.flush()

if cnt > 0:
    mean_iou  = sum_iou / float(cnt)
    pct_good  = 100.0 * cnt_iou_good / float(cnt)
    pct_stuck = 100.0 * cnt_stuck / float(cnt) 
    
    print(f"\n[Resumo dataset] imagens={cnt} | IoU médio (simplificado)={mean_iou:.3f} | %Imagens c/ IoU>0.5={pct_good:.1f}% | %stuck≈2.0={pct_stuck:.1f}%")

    summary = {
        "run_id": RUN_ID,
        "dataset_images": cnt,
        "mean_IoU_simplified": float(mean_iou),
        "pct_IoU_ge_0_5": float(pct_good),
        "pct_stuck_ge_approx_2": float(pct_stuck),
        "config": {
            "IN_SIZE": IN_SIZE, "IMG_SIZE": IMG_SIZE, "R_NORM": R_NORM,
            "RING_SAMPLES_COARSE": RING_SAMPLES_COARSE, "RING_SAMPLES_FINE": RING_SAMPLES_FINE,
            "W_PROBE_COARSE": W_PROBE_COARSE, "W_PROBE_FINE": W_PROBE_FINE,
            "MOMENTUM_BETA": MOMENTUM_BETA, "USE_GRAY_CODE": USE_GRAY_CODE,
            "POLICY": "SEQUENCIAL_SEARCH_EVAL_ONLY",
            "AUTO_EXTEND_STEPS": AUTO_EXTEND_STEPS,
            "EXTRA_STEPS_CAP_TRACE": EXTRA_STEPS_CAP_TRACE,
            "SEARCH_BUDGET": SEARCH_BUDGET, "INFER_STEPS_MULT": INFER_STEPS_MULT,
            "SEQUENTIAL_CONFIG": {
                "MAX_BAD_CANDIDATES_IN_A_ROW": MAX_BAD_CANDIDATES_IN_A_ROW,
                "STOP_LOSS_THR": STOP_LOSS_THR,
                "MIN_BLACK_PIXELS": MIN_BLACK_PIXELS
            }
        }
    }
    with open(RUN_JSONL_PATH, 'a', encoding='utf-8') as fsum:
        fsum.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(f"[Logs] JSONL salvo em: {RUN_JSONL_PATH}")
