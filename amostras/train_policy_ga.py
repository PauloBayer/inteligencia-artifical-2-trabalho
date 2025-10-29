# train_policy_ga.py — GA-trained binary quadrant policy (optimized + longer training + viz)
import os, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------
# Robust imports (run from repo root OR amostras/)
# ----------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE if (HERE / "gapy").exists() else HERE.parent

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from gapy.ga import gago  # your GA

# dataset helpers from your generator
try:
    from generate_bolas_dataset import (
        Circle, interior_fill_fraction, border_cut_fraction, best_match_iou
    )
except ModuleNotFoundError:
    here_str = str(HERE)
    if here_str not in sys.path:
        sys.path.insert(0, here_str)
    from generate_bolas_dataset import (
        Circle, interior_fill_fraction, border_cut_fraction, best_match_iou
    )

# ----------------------------
# Config
# ----------------------------
np.random.seed(42)

ROOT = (HERE / "dados") if (HERE / "dados").exists() else (REPO_ROOT / "amostras" / "dados")
ANN_PATH = ROOT / "annotations.jsonl"

SIZE    = 255     # image size on disk
R_MIN   = 8
R_MAX   = 40
T_STEPS = 20      # rollout steps per episode (<= 1r per step)

# GA settings (beefed up for better learning)
POP = 150
GEN = 150
MUT = 0.10
ELI = 4

# ----------------------------
# Data
# ----------------------------
with open(str(ANN_PATH), "r", encoding="utf-8") as fh:
    ANNS_ALL = [json.loads(l) for l in fh if l.strip()]
ANNS = [rec for rec in ANNS_ALL if rec.get("split") == "single"]  # curriculum: start single only

def load_gray(path):
    return np.array(Image.open(str(path)).convert("L"))

# ----------------------------
# Precomputed grids + FAST metrics (no ogrid inside loops)
# ----------------------------
YY, XX = np.ogrid[:SIZE, :SIZE]

def interior_fill_fraction_fast(img: np.ndarray, c: Circle) -> float:
    # img uint8: black=0, white=255
    dx = XX - c.cx
    dy = YY - c.cy
    mask = (dx*dx + dy*dy) <= (c.r * c.r)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    filled = int((img[mask] == 0).sum())
    return filled / float(total)

def quadrant_fill_fracs_fast(img: np.ndarray, c: Circle):
    dx = XX - c.cx
    dy = YY - c.cy
    mask_c = (dx*dx + dy*dy) <= (c.r * c.r)
    NW = (XX <= c.cx) & (YY <= c.cy)
    NE = (XX >= c.cx) & (YY <= c.cy)
    SW = (XX <= c.cx) & (YY >= c.cy)
    SE = (XX >= c.cx) & (YY >= c.cy)
    out = []
    for Q in (NW, NE, SW, SE):
        m = mask_c & Q
        cnt = int(m.sum())
        out.append(0.0 if cnt == 0 else int((img[m] == 0).sum()) / float(cnt))
    return tuple(out)  # qNW, qNE, qSW, qSE

# ----------------------------
# Binary net (7 -> 32 -> 11)
#   Inputs: [fill, cut, qNW, qNE, qSW, qSE, r_norm]
#   Outputs (bits):
#     b0 = sign(dx), b1 = sign(dy), b2 = grow/shrink
#     b3..b10 = 8-bit theta (0..255) -> alpha = |sin(2π*theta/255)|
# ----------------------------
class BinaryNet:
    def __init__(self, in_dim=7, hid=32, out_dim=11):
        self.in_dim, self.hid, self.out_dim = in_dim, hid, out_dim
        self.W1 = np.random.randint(0, 2, size=(in_dim, hid), dtype=np.uint8)
        self.b1 = np.random.randint(0, 2, size=(hid,), dtype=np.uint8)
        self.W2 = np.random.randint(0, 2, size=(hid, out_dim), dtype=np.uint8)
        self.b2 = np.random.randint(0, 2, size=(out_dim,), dtype=np.uint8)

    def n_bits(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_bits(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2]).astype(np.uint8)

    def set_bits(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        idx = 0
        nW1 = self.W1.size; nb1 = self.b1.size; nW2 = self.W2.size; nb2 = self.b2.size
        self.W1 = bits[idx:idx+nW1].reshape(self.W1.shape); idx += nW1
        self.b1 = bits[idx:idx+nb1]; idx += nb1
        self.W2 = bits[idx:idx+nW2].reshape(self.W2.shape); idx += nW2
        self.b2 = bits[idx:idx+nb2]

    def forward_bits(self, feat01):
        # Thresholded binary perceptrons
        h_lin = (feat01 @ self.W1) + self.b1
        h = (h_lin >= (self.in_dim / 2.0)).astype(np.uint8)
        o_lin = (h @ self.W2) + self.b2
        y = (o_lin >= (self.hid / 2.0)).astype(np.uint8)
        return y

# ----------------------------
# Action decoding + utilities
# ----------------------------
def bits_to_uint8(msb_bits):
    val = 0
    for bit in msb_bits:
        val = (val << 1) | int(bit)
    return val

def decode_action(bits11, r):
    b0, b1, b2 = int(bits11[0]), int(bits11[1]), int(bits11[2])  # signs and grow/shrink
    theta_bits = bits11[3:11]
    theta = bits_to_uint8(theta_bits)                       # 0..255
    alpha = abs(math.sin(2.0 * math.pi * theta / 255.0))    # 0..1
    step = max(0, int(round(alpha * r)))                    # allow zero-step (stabilizes when close)
    dx = +step if b0 == 1 else -step
    dy = +step if b1 == 1 else -step
    grow = (b2 == 1)
    return dx, dy, step, grow

def clamp_circle(c, size):
    cx = min(max(c.cx, c.r), size-1-c.r)
    cy = min(max(c.cy, c.r), size-1-c.r)
    r = max(1, min(c.r, size//2))
    return Circle(cx, cy, r)

# ----------------------------
# Episode rollout (fast metrics; tunable perimeter; optional curriculum)
# ----------------------------
def rollout(img, gt_circles, net: BinaryNet, size=SIZE, r_min=R_MIN, r_max=R_MAX,
            T=T_STEPS, record=False, n_perim=120, curriculum=False):
    if curriculum and gt_circles:
        # start near GT center (easier early-learning task)
        gx = int(np.mean([g.cx for g in gt_circles]))
        gy = int(np.mean([g.cy for g in gt_circles]))
        start_x = gx + int(np.random.randint(-20, 21))
        start_y = gy + int(np.random.randint(-20, 21))
        c = Circle(start_x, start_y, (r_min + r_max)//2)
        c = clamp_circle(c, size)
    else:
        c = Circle(size//2, size//2, (r_min + r_max)//2)

    trace = [c] if record else None
    steps_taken = 0
    for _ in range(T):
        steps_taken += 1
        fill = interior_fill_fraction_fast(img, c)
        cut  = border_cut_fraction(img, c, n_samples=n_perim)
        qNW, qNE, qSW, qSE = quadrant_fill_fracs_fast(img, c)
        feat = np.array([fill, cut, qNW, qNE, qSW, qSE, c.r/float(r_max)], dtype=np.float32)
        ybits = net.forward_bits(feat)
        dx, dy, step, grow = decode_action(ybits, c.r)
        nx, ny = c.cx + dx, c.cy + dy
        nr = c.r + step if grow else max(1, c.r - step)
        c = clamp_circle(Circle(nx, ny, nr), size)
        if record:
            trace.append(c)
        if fill > 0.985 and cut < 0.02:
            break
    iou, _ = best_match_iou(size=size, candidate=c, gt_circles=gt_circles)
    cutf   = border_cut_fraction(img, c, n_samples=max(n_perim, 360))
    return c, iou, cutf, steps_taken, (trace if record else None)

# ----------------------------
# GA Fitness (fixed mini-batch → less noise; stronger cut weight)
# ----------------------------
def make_fitness(train_records):
    proto = BinaryNet()
    nbits = proto.n_bits()

    # pick once per run (fixed mini-batch for stability)
    FIXED_IDX = np.random.choice(len(train_records), size=min(12, len(train_records)), replace=False)

    def ffit(bits):
        bb = np.asarray(bits, dtype=np.uint8)
        if bb.size != nbits:
            bb = np.resize(bb, nbits).astype(np.uint8)  # conservative fixup
        proto.set_bits(bb)

        loss_acc = 0.0
        for idx in FIXED_IDX:
            rec = train_records[idx]
            img = load_gray(ROOT / rec["file"])
            gt  = [Circle(**c) for c in rec["circles"]]
            # curriculum True during GA, fast-ish perimeter
            _, iou, cutf, steps, _ = rollout(img, gt, proto, record=False, n_perim=180, curriculum=True)
            # reward shaping: emphasize getting fully inside (low cut) early on
            loss = (1.0 - iou) + 0.8*cutf + 0.005*(steps / T_STEPS)
            loss_acc += loss
        return loss_acc / float(len(FIXED_IDX))

    return ffit, proto, nbits

# ----------------------------
# Visualization helpers
# ----------------------------
def _draw_circle(draw: ImageDraw.ImageDraw, c: Circle, color=(255,0,0), width=3):
    for w in range(width):
        bb = (c.cx - c.r - w, c.cy - c.r - w, c.cx + c.r + w, c.cy + c.r + w)
        draw.ellipse(bb, outline=color)

def save_overlay(img_gray: np.ndarray, pred: Circle, gts, out_path: Path, subtitle: str = ""):
    img = Image.fromarray(img_gray).convert("RGB")
    draw = ImageDraw.Draw(img)
    for g in gts:
        _draw_circle(draw, g, color=(0,200,0), width=2)  # GT in green
    _draw_circle(draw, pred, color=(255,0,0), width=3)   # Pred in red
    if subtitle:
        draw.text((6,6), subtitle, fill=(0,120,255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))

def save_trace_gif(img_gray: np.ndarray, trace, gts, out_path: Path, dur_ms=80):
    frames = []
    for step, c in enumerate(trace):
        frame = Image.fromarray(img_gray).convert("RGB")
        draw = ImageDraw.Draw(frame)
        for g in gts:
            _draw_circle(draw, g, color=(0,200,0), width=2)
        _draw_circle(draw, c, color=(255,0,0), width=3)
        draw.text((6,6), f"step {step}", fill=(0,120,255))
        frames.append(frame)
    frames[0].save(str(out_path), save_all=True, append_images=frames[1:], duration=dur_ms, loop=0)

# ----------------------------
# Train + Validate + Viz
# ----------------------------
def main():
    if len(ANNS) < 10:
        raise RuntimeError("Need at least ~10 single-ball images. Generate more with your script.")

    perm = np.random.permutation(len(ANNS))
    split = int(0.8 * len(ANNS))
    train_recs = [ANNS[i] for i in perm[:split]]
    valid_recs = [ANNS[i] for i in perm[split:]]

    ffit, net, nbits = make_fitness(train_recs)
    gaoptions = {
        "PopulationSize": POP,
        "Generations": GEN,
        "MutationFcn": MUT,
        "EliteCount": ELI,
        "InitialPopulation": None,
    }

    # Train
    x_best, popx, fitvals = gago(ffit, nbits, gaoptions)

    # optional 1-gen polish with final pop (same fitness)
    gaoptions_polish = {**gaoptions, "Generations": 1, "InitialPopulation": popx}
    x_best, popx, fitvals = gago(ffit, nbits, gaoptions_polish)

    net.set_bits(x_best.astype(np.uint8))

    # Validate (no curriculum; moderate perimeter)
    ious, cuts, steps_all = [], [], []
    for rec in valid_recs:
        img = load_gray(ROOT / rec["file"])
        gt  = [Circle(**c) for c in rec["circles"]]
        c, iou, cutf, steps, _ = rollout(img, gt, net, record=False, n_perim=180, curriculum=False)
        ious.append(iou); cuts.append(cutf); steps_all.append(steps)

    print(f"Validation mean IoU:        {float(np.mean(ious)):.4f}")
    print(f"Validation mean border_cut: {float(np.mean(cuts)):.4f}")
    print(f"Validation mean steps:      {float(np.mean(steps_all)):.2f}")

    # Visualization export (high quality)
    VIZ_DIR = REPO_ROOT / "amostras" / "viz"
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    rows = ["file,iou,border_cut,steps,pred_cx,pred_cy,pred_r"]
    showcase = valid_recs[: min(8, len(valid_recs))]

    for rec in showcase:
        img_path = ROOT / rec["file"]
        img = load_gray(img_path)
        gts = [Circle(**c) for c in rec["circles"]]

        pred, iou, cutf, steps, trace = rollout(img, gts, net, record=True, n_perim=360, curriculum=False)

        stem = Path(rec["file"]).stem
        save_overlay(
            img, pred, gts,
            VIZ_DIR / f"{stem}_overlay.png",
            subtitle=f"IoU={iou:.3f}  cut={cutf:.3f}  steps={steps}"
        )
        if trace is not None:
            save_trace_gif(img, trace, gts, VIZ_DIR / f"{stem}_trace.gif", dur_ms=80)

        rows.append(f"{rec['file']},{iou:.6f},{cutf:.6f},{steps},{pred.cx},{pred.cy},{pred.r}")

    with open(str(VIZ_DIR / "valid_metrics.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))

    print(f"\nSaved overlays/GIFs/metrics to: {VIZ_DIR}")
    print("  - *_overlay.png  (GT green, prediction red)")
    print("  - *_trace.gif    (rollout steps)")
    print("  - valid_metrics.csv")

if __name__ == "__main__":
    main()
