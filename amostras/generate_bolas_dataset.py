#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arquivo para gerar dataset de bolas pretas em fundo branco (255x255) + métricas auxiliares

As saídas vão ficar nas pastas:
- dados/images/single/*.png    (imagens com 1 bola)
- dados/images/multi/*.png     (imagens com 2 ou mais bolas)
- dados/annotations.jsonl      (um JSON por linha, com metadados)

Uso:
  python generate_bolas_dataset.py --out ./dados --single 100 --multi 100 \
    --size 255 --r-min 8 --r-max 40 --k-min 2 --k-max 6 --allow-overlap \
    --seed 42

Requisitos:
  pip install numpy pillow
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

# Representação básica
@dataclass(frozen=True)
class Circle:
    cx: int
    cy: int
    r: int

# Utilidades de geração
def rng_integers(rng: np.random.Generator, low: int, high: int) -> int:
    """Inteiro uniforme em [low, high], inclusivo para 'low' e exclusivo para 'high' (padrão numpy)."""
    return int(rng.integers(low, high))


def sample_circle_inside(size: int, r_min: int, r_max: int, rng: np.random.Generator) -> Circle:
    """Sorteia um círculo (cx, cy, r) tal que o disco fique totalmente dentro da imagem size×size."""
    r = rng_integers(rng, r_min, r_max + 1)
    low = r
    high = size - 1 - r
    if low > high:
        raise ValueError(
            f"Raio {r} não cabe em size={size}. Ajuste r_min/r_max para caber completamente."
        )
    cx = rng_integers(rng, low, high + 1)
    cy = rng_integers(rng, low, high + 1)
    return Circle(cx=cx, cy=cy, r=r)


def circles_overlap(c1: Circle, c2: Circle, min_gap: float = 0.0) -> bool:
    """Checa se dois círculos se sobrepõem considerando um espaçamento mínimo (gap)."""
    dx = c1.cx - c2.cx
    dy = c1.cy - c2.cy
    dist = math.hypot(dx, dy)
    return dist < (c1.r + c2.r + min_gap)


def sample_non_overlapping_circles(
    size: int,
    k: int,
    r_min: int,
    r_max: int,
    rng: np.random.Generator,
    min_gap: float = 0.0,
    max_tries: int = 10_000,
) -> List[Circle]:
    """Sorteia 'k' círculos não sobrepostos (com gap) totalmente dentro da imagem."""
    circles: List[Circle] = []
    tries = 0
    while len(circles) < k and tries < max_tries:
        candidate = sample_circle_inside(size, r_min, r_max, rng)
        if all(not circles_overlap(candidate, c, min_gap=min_gap) for c in circles):
            circles.append(candidate)
        tries += 1
    if len(circles) < k:
        raise RuntimeError(
            f"Falha ao alocar {k} círculos sem sobreposição após {max_tries} tentativas. "
            f"Tente reduzir k, aumentar size ou diminuir r_max/min_gap."
        )
    return circles


def sample_circles(
    size: int,
    k: int,
    r_min: int,
    r_max: int,
    rng: np.random.Generator,
    allow_overlap: bool,
    min_gap: float = 0.0,
) -> List[Circle]:
    """Sorteia 'k' círculos (permitindo ou não sobreposição)."""
    if allow_overlap:
        return [sample_circle_inside(size, r_min, r_max, rng) for _ in range(k)]
    else:
        return sample_non_overlapping_circles(
            size=size, k=k, r_min=r_min, r_max=r_max, rng=rng, min_gap=min_gap
        )

# Rasterização e I/O de imagens
def draw_circles_to_array(size: int, circles: List[Circle]) -> np.ndarray:
    """
    Cria uma matriz (H, W) uint8 com fundo branco (255) e bolas pretas (0).
    Overlaps continuam pretos.
    """
    img = np.full((size, size), 255, dtype=np.uint8)  # demorei, mas achei, fundo branco aqui
    if not circles:
        return img
    yy, xx = np.ogrid[:size, :size]
    for c in circles:
        mask = (xx - c.cx) ** 2 + (yy - c.cy) ** 2 <= c.r ** 2
        img[mask] = 0
    return img


def save_image_gray(path: Path, arr: np.ndarray) -> None:
    """Salva a matriz uint8 (0..255) como PNG em escala de cinza ('L')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(str(path), format="PNG", optimize=True)


# Métricas para AG/RNA
def _circle_interior_mask(size: int, circle: Circle) -> np.ndarray:
    """Máscara booleana dos pixels estritamente dentro do círculo."""
    yy, xx = np.ogrid[:size, :size]
    return (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2 <= circle.r ** 2


def _circle_perimeter_samples(circle: Circle, n_samples: int = 360) -> List[Tuple[int, int]]:
    """
    Amostra 'n_samples' pontos inteiros na circunferência (arredondando).
    Útil para estimar 'border_cut_fraction'.
    """
    pts: List[Tuple[int, int]] = []
    for i in range(n_samples):
        theta = (2.0 * math.pi) * (i / n_samples)
        x = int(round(circle.cx + circle.r * math.cos(theta)))
        y = int(round(circle.cy + circle.r * math.sin(theta)))
        pts.append((x, y))
    return pts


def interior_fill_fraction(img: np.ndarray, circle: Circle) -> float:
    """
    Fração de pixels do interior do 'circle' que são pretos (0).
    Retorna [0..1]. Quanto mais próximo de 1, mais o candidato cobre área preta.
    """
    h, w = img.shape
    mask = _circle_interior_mask(size=h, circle=circle)
    if np.count_nonzero(mask) == 0:
        return 0.0
    # Preto é 0 → '== 0' indica interior preenchido por bola(s)
    filled = np.count_nonzero(img[mask] == 0)
    return float(filled) / float(np.count_nonzero(mask))


def border_cut_fraction(img: np.ndarray, circle: Circle, n_samples: int = 360) -> float:
    """
    Fração de amostras na circunferência do 'circle' que caem em pixels pretos (0).
    Interpretação:
      - ≈0.0 → a borda do círculo candidato não corta regiões pretas (está no 'miolo' ou no fundo branco)
      - >0.0 → a borda toca/corta a(s) bola(s) (indicando ajuste na posição/raio)
    """
    h, w = img.shape
    pts = _circle_perimeter_samples(circle, n_samples=n_samples)
    total = 0
    black = 0
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            total += 1
            if img[y, x] == 0:
                black += 1
    if total == 0:
        return 0.0
    return black / total


def iou_with_mask(candidate_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU binária entre máscara do candidato e ground-truth."""
    inter = np.count_nonzero(candidate_mask & gt_mask)
    union = np.count_nonzero(candidate_mask | gt_mask)
    return (inter / union) if union > 0 else 0.0


def best_match_iou(
    size: int, candidate: Circle, gt_circles: List[Circle]
) -> Tuple[float, Optional[int]]:
    """
    IoU do candidato contra a melhor bola ground-truth (retorna (iou, idx_gt)).
    Útil quando sua RNA tenta encontrar uma bola de cada vez.
    """
    if not gt_circles:
        return 0.0, None
    cand_mask = _circle_interior_mask(size=size, circle=candidate)
    best_iou = 0.0
    best_idx = None
    for idx, gt in enumerate(gt_circles):
        gt_mask = _circle_interior_mask(size=size, circle=gt)
        iou = iou_with_mask(cand_mask, gt_mask)
        if iou > best_iou:
            best_iou, best_idx = iou, idx
    return best_iou, best_idx

# Geração da amostra
def write_jsonl_line(fh, obj: Dict[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

def make_annotations_record(
    rel_path: str,
    split: str,
    size: int,
    circles: List[Circle],
) -> Dict[str, Any]:
    return {
        "file": rel_path.replace("\\", "/"),
        "split": split,  # "single" ou "multi"
        "size": size,
        "n_bolas": len(circles),
        "circles": [asdict(c) for c in circles],
    }

def build_dataset(
    out_dir: Path,
    size: int,
    n_single: int,
    n_multi: int,
    r_min: int,
    r_max: int,
    k_min_multi: int,
    k_max_multi: int,
    allow_overlap: bool,
    min_gap: float,
    seed: int,
) -> None:
    out_dir = Path(out_dir)
    img_single_dir = out_dir / "images" / "single"
    img_multi_dir = out_dir / "images" / "multi"
    ann_path = out_dir / "annotations.jsonl"

    rng = np.random.default_rng(seed)

    # Garante as pastas
    img_single_dir.mkdir(parents=True, exist_ok=True)
    img_multi_dir.mkdir(parents=True, exist_ok=True)

    with ann_path.open("w", encoding="utf-8") as fh:
        # SINGLE
        for i in range(n_single):
            circles = sample_circles(
                size=size,
                k=1,
                r_min=r_min,
                r_max=r_max,
                rng=rng,
                allow_overlap=True,  # com 1 bola, tanto faz
                min_gap=0.0,
            )
            arr = draw_circles_to_array(size=size, circles=circles)
            fname = f"single_{i:04d}.png"
            fpath = img_single_dir / fname
            save_image_gray(fpath, arr)
            rec = make_annotations_record(
                rel_path=str(fpath.relative_to(out_dir)),
                split="single",
                size=size,
                circles=circles,
            )
            write_jsonl_line(fh, rec)

        # MULTI
        for i in range(n_multi):
            k = rng_integers(rng, k_min_multi, k_max_multi + 1)
            circles = sample_circles(
                size=size,
                k=k,
                r_min=r_min,
                r_max=r_max,
                rng=rng,
                allow_overlap=allow_overlap,
                min_gap=min_gap,
            )
            arr = draw_circles_to_array(size=size, circles=circles)
            fname = f"multi_{i:04d}.png"
            fpath = img_multi_dir / fname
            save_image_gray(fpath, arr)
            rec = make_annotations_record(
                rel_path=str(fpath.relative_to(out_dir)),
                split="multi",
                size=size,
                circles=circles,
            )
            write_jsonl_line(fh, rec)

    print(f"✔ Dataset pronto em: {out_dir}")
    print(f"  - Images single: {img_single_dir}")
    print(f"  - Images multi : {img_multi_dir}")
    print(f"  - Annotations  : {ann_path}")

def demo_metrics_example(
    out_dir: Path, img_rel_path: str, candidate: Circle
) -> Dict[str, Any]:
    """
    Carrega uma imagem gerada e calcula métricas da candidata lá no arquivo JSON.
    """
    img_path = Path(out_dir) / img_rel_path
    img = np.array(Image.open(img_path).convert("L"))
    # Carrega GT do annotations.jsonl
    ann_path = Path(out_dir) / "annotations.jsonl"
    gt_circles: List[Circle] = []
    with ann_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["file"] == img_rel_path.replace("\\", "/"):
                gt_circles = [Circle(**c) for c in rec["circles"]]
                break

    metrics = {
        "interior_fill_fraction": interior_fill_fraction(img, candidate),
        "border_cut_fraction": border_cut_fraction(img, candidate, n_samples=720),
    }
    iou, best_idx = best_match_iou(size=img.shape[0], candidate=candidate, gt_circles=gt_circles)
    metrics["best_match_iou"] = iou
    metrics["best_match_gt_idx"] = best_idx
    return metrics

# Os argumentos e tals do CLI vão aqui
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gerador de dataset de bolas pretas 255x255 + métricas auxiliares."
    )
    p.add_argument("--out", type=str, default="./dados", help="Diretório de saída do dataset.")
    p.add_argument("--size", type=int, default=255, help="Tamanho da imagem (size×size).")
    p.add_argument("--single", type=int, default=100, help="Número de imagens com 1 bola.")
    p.add_argument("--multi", type=int, default=100, help="Número de imagens com múltiplas bolas.")
    p.add_argument("--k-min", type=int, default=2, help="Mínimo de bolas no modo multi.")
    p.add_argument("--k-max", type=int, default=5, help="Máximo de bolas no modo multi.")
    p.add_argument("--r-min", type=int, default=8, help="Raio mínimo das bolas.")
    p.add_argument("--r-max", type=int, default=40, help="Raio máximo das bolas.")
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Permite sobreposição entre bolas no split multi.",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Gap mínimo entre bordas quando overlap NÃO é permitido (pixels).",
    )
    p.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade.")
    p.add_argument(
        "--demo-metrics",
        action="store_true",
        help="Após gerar o dataset, roda um exemplo de métricas em uma imagem.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    build_dataset(
        out_dir=out_dir,
        size=args.size,
        n_single=args.single,
        n_multi=args.multi,
        r_min=args.r_min,
        r_max=args.r_max,
        k_min_multi=args.k_min,
        k_max_multi=args.k_max,
        allow_overlap=bool(args.allow_overlap),
        min_gap=float(args.min_gap),
        seed=args.seed,
    )

    if args.demo_metrics:
        # Pega uma imagem aleatória do split single para demonstrar
        img_rel = f"images/single/single_{0:04d}.png"
        candidate = Circle(cx=args.size // 2, cy=args.size // 2, r=max(args.r_min, 12))
        m = demo_metrics_example(out_dir, img_rel, candidate)
        print("Exemplo de métricas para candidato em", img_rel)
        print(json.dumps(m, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
