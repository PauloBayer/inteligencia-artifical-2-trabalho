# ------------------------------------------------------------
# No momento, este é um controlador de uma única bola, com a RNA decidindo 
# a magnitude de mudança do raio.
#
# Ideia geral:
# - Temos imagens de fundo claro com artefatos pretos circulares.
# - Uma rede neural simples sugere ações:
#     * direção de movimento do centro (x/y),
#     * e direção + magnitude para ajustar o raio do círculo candidato.
# - Um Algoritmo Genético (AG) busca diretamente pesos que minimizam uma função de perda
#   (loss) em cima de métricas geométricas simples (fração de pixels
#   pretos em anéis, “fill” dentro do círculo, etc), fazendo com que a RNA aprenda.
# - O controlador executa passos (movimento + ajuste de raio),
#   avalia, guarda o melhor e aplica paradas inteligentes.
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
# 0) Configuração global do experimento
# ============================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
rng = np.random.default_rng(GLOBAL_SEED)

IN_SIZE  = 28      # Resolução reduzida que vai para a RNA (entrada de baixa dimensão)
IMG_SIZE = 255     # Tamanho do canvas quadrado em pixels (255x255)
R_NORM   = int(math.ceil(math.hypot(IMG_SIZE - 1, IMG_SIZE - 1)))  # Normalizador pro raio (~360)

DATA_ROOT = "amostras/dados"
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, "annotations.jsonl")

# --- "Binarização" simples no full-res (tons de cinza) para decidir o que é "preto" ---
BLACK_THR = 64          # Intensidades <= 64 contam como preto

# --- Parada por “acerto perfeito” (ajuste fino quando o anel encaixa) ---
STOP_ON_PERFECT      = True
PERFECT_INNER_FRAC   = 0.995  # Fração de preto esperada logo dentro da borda
PERFECT_OUTER_FRAC   = 0.005  # Fração de preto tolerada logo fora da borda
PERFECT_THICKNESS    = 1      # Espessura (em amostragens de anel) usada na avaliação "perfeita"

# Parada por IoU de máscara (robusta a aliasing de borda) ---
# IoU (Intersection over Union) entre o círculo rasterizado e a máscara de pixels pretos.
MASK_IOU_STOP_ENABLE = True
MASK_IOU_STOP_THR    = 0.985   # Parar quando a concordância com a máscara for >= 98.5%

# --- Flags de overlay em GIF ---
SHOW_GT_IN_GIF       = True   # Desenha círculo do ground truth (GT) no GIF, se disponível
PRED_ON_TOP          = True   # Predição por cima do GT (ou vice-versa)

# --- Recorte/trim de GIF quando ocorre parada ---
TRIM_GIF_AT_STOP     = True
KEEP_SNAP_AFTER_STOP = True   # Mantém um frame extra de “snap refine” se houver

# --- HUD (texto sobre o GIF) ---
HUD_TEXT_COLOR   = (255, 255, 255)
HUD_STROKE_COLOR = (0, 0, 0)
HUD_BG_RGBA      = None
HUD_FONT_PATH    = None
HUD_FONT_SIZE    = 10
HUD_PAD          = 2
HUD_POS          = (5, 5)

# --- Anti-travamento em quadros totalmente brancos (sem sinal) ---
# se estamos “no nada”, aumentamos passo, injetamos jitter e
# eventualmente damos super saltos para sair de regiões puramente brancas.
WHITE_STREAK_GROWTH   = 0.80
WHITE_JITTER_PX       = 3
WHITE_JITTER_CLAMP    = 24
WHITE_SUPERJUMP_EVERY = 10
WHITE_SUPERJUMP_PX    = IMG_SIZE // 2

# --- Crescimento do raio quando só vemos branco por muito tempo ---
# Isso evita ficar varrendo com um raio pequeno sem nunca tocar borda preta.
WHITE_RADIUS_GROW_AFTER    = 20
WHITE_RADIUS_GROW_EVERY    = 4
WHITE_RADIUS_GROW_MAX_PX   = 6
WHITE_RADIUS_REQUIRE_BIT   = True
WHITE_RADIUS_RESET_STREAK  = False

# --- Amansadores de overshoot quando aproximando da borda ---
# Ganhos diferentes longe/perto da borda e tetos de passo.
APPROACH_STEP_GAIN_FAR     = 0.60
APPROACH_STEP_GAIN_NEAR    = 0.35
APPROACH_NEAR_FRAC_OF_R    = 0.75
APPROACH_STEP_MAX_PX       = 32
BORDER_STEP_MAX_PX         = 24
MOMENTUM_DAMP_ON_MODE_CHANGE = 0.25  # Amortece momento quando troca de "modo" (evita pular a borda)

# --- Heurísticas mínimas para o sinal do raio (só o sinal, já que a magnitude vem da RNA) ---
# Observa anéis imediatamente fora/dentro da borda para decidir expandir/contrair.
EPS_OUTER_EXPAND = 0.02  # >2% preto no anel externo sugere expandir (estamos cortando a borda)
EPS_INNER_SHRINK = 0.40  # anel interno <40% preto (e interior com preto) sugere encolher

# --- Tetos de variação de raio dependentes da distância ---
RAD_CAP_NEAR_PX = 1   # Quando “perto” da borda, mudar pouco o raio (evita passa-passa)
RAD_CAP_FAR_PX  = 8   # Quando “longe” pode ajustar mais rápido

# --- Métricas/avaliação de IoU com GT (quando GT existe, p.ex. no modo com traço/GIF) ---
IOU_EVAL_STOP      = True
IOU_EVAL_THRESH    = 1.0
IOU_CHASE_ENABLE   = True
IOU_IMPROVE_DELTA  = 0.001
IOU_STEPS_BONUS    = 64
IOU_EXTRA_CAP      = 2000

# --- Quebra de “ping-pong” (alternância 2-ciclos) ---
# Se o sistema começa a alternar expandir/encolher num mesmo par de estados, interrompemos.
PINGPONG_TOL_PX         = 1     # Tolerância para considerar posição/raio “iguais”
PINGPONG_REQUIRE_MODE   = True  # Exigir alternância entre modos expandir<->encolher
PINGPONG_ONLY_WHEN_NEAR = True  # Só ativa quando estamos “perto da borda” (via testes de anéis)

# ============================================================
# 1) Hiperparâmetros e presets (perfis de busca)
# ============================================================
CTRL_STEPS_COARSE = 8    # Nº de passos de controle na fase “grossa”
CTRL_STEPS_FINE   = 12   # Nº de passos na fase de refinamento
PATIENCE_STEPS    = 4    # Paciência da parada precoce por falta de melhora
WARMUP_STEPS      = 3    # Passos iniciais sem punir falta de melhora
IMPROVE_EPS       = 1e-6 # Margem mínima para considerar que “melhorou”

RING_SAMPLES_COARSE = 128  # Amostras ao longo do anel (fase grossa)
RING_SAMPLES_FINE   = 256  # Amostras ao longo do anel (fase fina)

# Raios de prova para métricas auxiliares (ver função _make_probe_list)
PROBE_R_COARSE   = 48
PROBE_R_FINE     = 64
PROBE_THICKNESS  = 2
W_PROBE_COARSE   = 0.35
W_PROBE_FINE     = 0.10

# Limiares para considerar “borda boa”: dentro bem preto e fora bem claro
TH_INNER = 0.90
TH_OUTER = 0.10

# Piso de passo para o raio (evita mudar 0 px quando rede sugere algo muito pequeno)
RAD_STEP_FLOOR_PX   = 1
TH_EXPAND_INNER     = 0.85  # (legado, manti só para compatibilidade)
TH_EXPAND_OUTER     = 0.50  # (legado)

# Extensão automática do orçamento de passos quando há sinal útil
AUTO_EXTEND_STEPS       = True
EXTRA_STEPS_HAS_SIGNAL  = 8
EXTRA_STEPS_BORDER      = 6
EXTRA_STEPS_REFINE      = 4
EXTRA_STEPS_CAP         = 24
EXTRA_STEPS_CAP_TRACE   = 2000

# Parâmetros do AG (população, gerações, mutação, elitismo)
GA_POP_COARSE   = 120
GA_GENS_COARSE  = 22
GA_MUT_COARSE   = 0.90
GA_POP_POLISH   = GA_POP_COARSE
GA_GENS_POLISH  = 6
GA_MUT_POLISH   = 0.90
ELITE_COUNT     = 2

# --- Encolhimento mais rápido (a RNA ainda decide a magnitude base) ---
# se o anel interno indica “estamos grandes demais”, multiplicar o passo
# de encolhimento para convergir mais ágil, com tetos por proximidade.
SHRINK_GAIN_MIN      = 1.25
SHRINK_GAIN_MAX      = 3.0
SHRINK_CAP_NEAR_PX   = 8
SHRINK_CAP_FAR_PX    = 14

# Permitimos círculos parcialmente fora da imagem (cálculos clipam no canvas)
ALLOW_PARTIAL_CIRCLE = True

# Raio máximo (diagonal do canvas é um limite tranquilo)
R_EXT_MAX = int(math.ceil(math.hypot(IMG_SIZE, IMG_SIZE)))  # ~360 para 255x255

import os as _os
EARLY_STOP = False
SEARCH_BUDGET = _os.getenv("NN_BUDGET", "thorough")
PRESETS = {
    # Perfis de orçamento de busca (menos passos = mais rápido, menos precisão)
    "fast":      {"CTRL_STEPS_COARSE": 6,  "CTRL_STEPS_FINE": 10,  "PATIENCE_STEPS": 2,  "EXTRA_STEPS_CAP": 12},
    "balanced":  {"CTRL_STEPS_COARSE": 8,  "CTRL_STEPS_FINE": 12,  "PATIENCE_STEPS": 4,  "EXTRA_STEPS_CAP": 24},
    "thorough":  {"CTRL_STEPS_COARSE": 24, "CTRL_STEPS_FINE": 64,  "PATIENCE_STEPS": 8,  "EXTRA_STEPS_CAP": 256},
    "max":       {"CTRL_STEPS_COARSE": 48, "CTRL_STEPS_FINE": 128, "PATIENCE_STEPS": 12, "EXTRA_STEPS_CAP": 2000},
}
_p = PRESETS.get(SEARCH_BUDGET, PRESETS["max"]) # Botei em max aqui
CTRL_STEPS_COARSE = _p["CTRL_STEPS_COARSE"]
CTRL_STEPS_FINE   = _p["CTRL_STEPS_FINE"]
PATIENCE_STEPS    = _p["PATIENCE_STEPS"]
EXTRA_STEPS_CAP   = _p["EXTRA_STEPS_CAP"]

# No modo com traço (trace/GIF), deixamos vagar mais (mais passos) por padrão
INFER_STEPS_MULT = int(_os.getenv("NN_INFER_MULT", "3"))

# ============================================================
# 1.2) Inicialização / decodificação / movimento
# ============================================================
# Estratégias de chute inicial (posição/raio) antes de começar a se mover, com o head que a RNA aprendeu
INIT_POLICY            = "random_then_nn" # Política de início
INIT_RANDOM_R_MIN_PX   = 4
INIT_RANDOM_R_MAX_FRAC = 0.45
INIT_NN_STEPS          = 1
INIT_STEP_ABS_SCALE    = IMG_SIZE
USE_INIT_HEAD          = False  # Se True, a RNA tem 3 saídas extras para sugerir (cx, cy, r) diretamente

# Decodificação das ações (rede emite bits): opção de usar Gray code para robustez
USE_GRAY_CODE     = True
MOVE_GAMMA        = 0.85  # suavização da fração de passo (não-linearidade)
RAD_GAMMA         = 0.85

# Momento (inércia) no movimento do centro
MOMENTUM_BETA     = 0.6
DEADZONE_PX       = 0     # Pequena zona morta opcional

# Estratégias quando estamos “no branco” (sem sinal de borda/interior)
WHITE_USE_NN_STEP      = True
WHITE_STEP_ABS_MAX     = IMG_SIZE // 2
WHITE_STUCK_PATIENCE   = 2
WHITE_JUMP_PX          = IMG_SIZE // 3

# Infra de logs/resultados e GIFs
RUNS_DIR          = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
RUN_ID            = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_JSONL_PATH    = os.path.join(RUNS_DIR, f"run_{RUN_ID}.jsonl")

MAKE_GIFS         = True
GIFS_DIR          = os.path.join(RUNS_DIR, "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)
GIF_SCALE         = 2
GIF_DURATION_MS   = 120
GIF_TAIL_HOLD_MS  = 5000
GIF_TAIL_COMPAT_DUPLICATE  = False
GIF_TAIL_DUPLICATE_EACH_MS = 250
GIF_LIMIT         = None

# ============================================================
# 2) Entrada/Saída de dados (load de imagem)
# ============================================================
def load_image_small_bin(file_path, out_size=IN_SIZE):
    """
    Lê a imagem, converte para tons de cinza, reamostra com vizinho mais próximo
    para out_size x out_size e binariza em 0/1 (1 = “preto”). Essa é a entrada
    comprimida que vai para a RNA, junto com (cx, cy, r) normalizados.
    """
    img = Image.open(file_path).convert('L').resize((out_size, out_size), Image.NEAREST)
    arr = np.array(img)
    return np.where(arr < 128, 1.0, 0.0).astype(np.float32)

def load_image_full_gray(file_path):
    """Imagem original em escala de cinza, resolução completa (para métricas e desenho)."""
    return np.array(Image.open(file_path).convert('L'))

# ============================================================
# 3) Amostragem em anéis e métricas geométricas
# ============================================================
def _precompute_trig(n_samples):
    """
    Pré-computa cos/sen igualmente espaçados em [0, 2π) para gerar coordenadas de anéis.
    Evita recalcular trigonometria a cada passo de controle.
    """
    theta = (2.0 * np.pi) * (np.arange(n_samples, dtype=np.float32) / float(n_samples))
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

# Tabelas trigonométricas para fase grossa e fina
COS_COARSE, SIN_COARSE = _precompute_trig(RING_SAMPLES_COARSE)
COS_FINE,   SIN_FINE   = _precompute_trig(RING_SAMPLES_FINE)

def _ring_coords(cx, cy, rr, cos_tab, sin_tab, size):
    """
    Constrói as coordenadas (x,y) inteiras de um anel de raio rr ao redor de (cx, cy).
    Descarta amostras fora da imagem (clipping por máscara de validade).
    """
    rr = int(max(1, abs(int(round(rr)))))
    xs = np.rint(cx + rr * cos_tab).astype(np.int32)
    ys = np.rint(cy + rr * sin_tab).astype(np.int32)
    valid = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    return xs[valid], ys[valid]

def _ring_fraction_vec(img255, cx, cy, r, delta, cos_tab, sin_tab):
    """
    Mede fração de pixels pretos em um anel deslocado: r+delta.
    Serve para estimar quão bom está o acerto de borda (dentro preto, fora claro).
    """
    size = img255.shape[0]
    rr = int(round(abs((r + delta) if r != 0 else delta)))
    xs = np.rint(cx + rr * cos_tab).astype(np.int32)
    ys = np.rint(cy + rr * sin_tab).astype(np.int32)
    total = xs.size
    if total == 0:
        return 0.0
    valid = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    if not np.any(valid):
        black = 0  # fora da imagem conta como “branco”
    else:
        vals = img255[ys[valid], xs[valid]]
        black = int(np.count_nonzero(vals <= BLACK_THR))
    return black / float(total)

def _ring_fraction_thick(img255, cx, cy, r, delta_center, thickness, cos_tab, sin_tab):
    """
    Versão “espessa”: média de fração de preto em vários anéis vizinhos,
    centrados em (r + delta_center). Reduz sensibilidade a aliasing.
    """
    if thickness <= 0:
        return _ring_fraction_vec(img255, cx, cy, r, delta_center, cos_tab, sin_tab)
    acc = 0.0; cnt = 0
    for u in range(-thickness, thickness + 1):
        acc += _ring_fraction_vec(img255, cx, cy, r, delta_center + u, cos_tab, sin_tab)
        cnt += 1
    return acc / float(cnt) if cnt > 0 else 0.0

def _border_cut_vec(img255, cx, cy, r, cos_tab, sin_tab):
    """
    Mede fração de preto exatamente na borda (anel em delta=0).
    Útil para detectar “corte” da borda (quando raio não está alinhado).
    """
    return _ring_fraction_vec(img255, cx, cy, r, delta=0, cos_tab=cos_tab, sin_tab=sin_tab)

def _circle_mask(size, cx, cy, r):
    """Máscara booleana de um disco (<= r^2) dentro do canvas size x size."""
    yy, xx = np.ogrid[:size, :size]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

def interior_fill_fraction(img255, cx, cy, r):
    """
    Fração de pixels pretos dentro do círculo: 1.0 se interior está todo preto.
    Essa métrica puxa o raio para “abraçar” a região preta sem invadir fundo branco.
    """
    mask = _circle_mask(img255.shape[0], cx, cy, r)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    filled = int(np.count_nonzero(img255[mask] <= BLACK_THR))
    return filled / float(area)

def iou_circle(size, c1, c2):
    """
    IoU entre dois círculos (métricas de avaliação com GT).
    Intersecção / União das máscaras discretas.
    """
    m1 = _circle_mask(size, c1[0], c1[1], c1[2])
    m2 = _circle_mask(size, c2[0], c2[1], c2[2])
    inter = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    return (inter / union) if union > 0 else 0.0

# ============================================================
# 3.b) IoU com máscara discreta (fundo preto) e snap refine
# ============================================================
def black_mask(img255, thr=BLACK_THR):
    """Máscara booleana: True onde pixel <= thr (considerado preto)."""
    return (img255 <= thr)

def iou_circle_vs_mask(img255, cx, cy, r, thr=BLACK_THR):
    """
    IoU entre o círculo candidato e a máscara de “pretos” da imagem.
    Para eixos robustos sem GT: se IoU ficar muito alto, já é um bom ponto de parada.
    """
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
    Pequena busca local (±dxy em x/y e ±dr no raio) para “encaixar” melhor o círculo.
    Critério: maximizar IoU contra a máscara preta. Em empates, preferir raio menor
    (evita círculo grande demais que pega ruído).
    """
    size = img255.shape[0]
    best_score = iou_circle_vs_mask(img255, cx, cy, r, thr)
    best = (int(cx), int(cy), int(r))
    for dy in range(-dxy, dxy + 1):
        for dx in range(-dxy, dxy + 1):
            cx2 = int(np.clip(cx + dx, 0, size - 1))
            cy2 = int(np.clip(cy + dy, 0, size - 1))
            rmax2 = r_fit_for_center(size, cx2, cy2)
            for dr_ in range(-dr, dr + 1):
                r2 = int(np.clip(r + dr_, 1, rmax2))
                s = iou_circle_vs_mask(img255, cx2, cy2, r2, thr)
                if (s > best_score) or (abs(s - best_score) < 1e-12 and prefer_smaller_radius and r2 < best[2]):
                    best_score = s
                    best = (cx2, cy2, r2)
    return best, best_score

# ============================================================
# 3.c) Funções de Mascaramento e Verificação para Busca Sequencial
# ============================================================
def array_to_image_small_bin(img_array_full, out_size=IN_SIZE):
    """
    Converte um array NumPy (imagem full-res) para a entrada binária 28x28 da RNA.
    Isso é necessário para re-alimentar a RNA com a imagem mascarada.
    """
    # Converte o array full-res para um objeto Image da PIL
    # O Image.fromarray assume que o input é L (escala de cinza)
    img_pil = Image.fromarray(img_array_full.astype(np.uint8)) 
    
    # Redimensiona com Vizinho Mais Próximo (como na função original)
    img_small = img_pil.resize((out_size, out_size), Image.NEAREST).convert('L')
    arr = np.array(img_small)
    
    # Binariza (como na função original: < 128 é preto/1.0, >= 128 é branco/0.0)
    return np.where(arr < 128, 1.0, 0.0).astype(np.float32)

def mask_circle_in_image(img255, cx, cy, r, val=255):
    """
    Apaga uma bola encontrada, setando os pixels dentro do círculo para branco (255)
    na imagem full-res (img255).
    """
    size = img255.shape[0]
    # Reutiliza a função _circle_mask (que deve estar definida antes desta seção)
    mask = _circle_mask(size, cx, cy, r)
    img255[mask] = val # Seta a área para branco
    return img255

def check_for_black(img255, thr=BLACK_THR):
    """Verifica se ainda restam pixels pretos (além do limiar BLACK_THR) na imagem."""
    # Retorna True se houver algum pixel com intensidade <= BLACK_THR
    return np.any(img255 <= thr)

# ============================================================
# 4) Probes auxiliares + função de perda (loss)
# ============================================================
def _make_probe_list(base_r, img_size):
    """
    Cria uma pequena lista de raios de prova (pontos de checagem externos) para
    penalizar configurações absurdas. Isso ajuda a RNA/AG a não cair em mínimos ruins.
    """
    lst = [int(base_r), int(2*base_r), int(3*base_r)]
    max_r = int(0.45 * img_size)
    return [r for r in lst if r >= 2 and r <= max_r] or [max(2, min(lst))]

def _probe_max_thick(img255, cx, cy, probe_r_list, thickness, cos_tab, sin_tab):
    """
    Em cada raio de prova, medimos fração de preto num anel espesso, pegamos o máximo.
    A loss inclui um termo (1 - max_probe)^2 para incentivar que exista um “anel preto”
    razoável no entorno. Isso protege contra soluções vazias.
    """
    if not probe_r_list:
        return 0.0
    vals = [
        _ring_fraction_thick(img255, cx, cy, r=0, delta_center=pr, thickness=thickness,
                             cos_tab=cos_tab, sin_tab=sin_tab)
        for pr in probe_r_list
    ]
    return max(vals) if vals else 0.0

def make_metrics_loss(img255, cos_tab, sin_tab, cache_dict, probe_r_list, weights=None, w_probe=0.20):
    """
    Constrói e retorna uma função metrics_loss(cx, cy, r) -> float.
    Essa loss combina:
      - fill (interior preto),
      - inner ring preto,
      - outer ring claro,
      - cut (borda “cortando”),
      - probe (anel de prova).
    Quanto menor, melhor. O AG otimiza os pesos da RNA para minimizar essa loss.
    """
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
# 5) MLP (Rede Neural Artificial)
# ============================================================
class NeuralNetwork:
    """
    Uma RNA feed-forward pequena:
      - camadas ocultas com sigmoid,
      - saída com sigmoid,
      - pesos/bias inicializados com N(0, 0.01).
    O AG busca diretamente os melhores pesos para minimizar a loss.
    """
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
        """
        Sigmoid numérica estável: evita overflow/underflow separando casos x>=0 e x<0.
        """
        x = np.asarray(x, dtype=np.float32)
        z = np.empty_like(x, dtype=np.float32)
        pos = x >= 0; neg = ~pos
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg]); z[neg] = ex / (1.0 + ex)
        return z

    def forward(self, x):
        """Passagem direta: aplica camadas ocultas + camada de saída com sigmoid."""
        a = x
        for i in range(len(self.hidden_sizes)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
        out = self.sigmoid(np.dot(a, self.weights[-1]) + self.biases[-1])
        return out

    def get_weights(self):
        """Achata todos os pesos/bias em um único vetor 1D (para o AG manipular)."""
        vec = []
        for w, b in zip(self.weights, self.biases):
            vec.append(w.flatten()); vec.append(b)
        return np.concatenate(vec).astype(np.float32)

    def set_weights(self, vector):
        """Reconstrói pesos/bias a partir de um vetor 1D (saída do decodificador do AG)."""
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            n_w = int(np.prod(w_shape))
            self.weights[i] = vector[idx:idx+n_w].reshape(w_shape).astype(np.float32); idx += n_w
            n_b = int(np.prod(b_shape))
            self.biases[i] = vector[idx:idx+n_b].astype(np.float32); idx += n_b

# ============================================================
# A RNA emite 22 bits de ação principais, opcionalmente mais 3 floats para init head
# Esses 22 bits são decodificados por decode_actions(...) em:
#   - direção do movimento do centro (sx, sy) ∈ {−1, 0, +1}
#   - magnitude do passo de movimento do centro (move_step)
#   - sinal de variação do raio (sr) ∈ {−1, 0, +1}
#   - magnitude da variação do raio (rad_step)
#
# Mapa bit a bit (índices 0..21):
#   [0]  -> x+     : tenta mover o centro para +X (direita)
#   [1]  -> x-     : tenta mover o centro para −X (esquerda)
#   [2]  -> y+     : tenta mover o centro para +Y (baixo)
#   [3]  -> y-     : tenta mover o centro para −Y (cima)
#   [4..11]  move_u8 (8 bits, LSB-first):
#            magnitude do passo de movimento do centro
#            (pode vir em Gray code se USE_GRAY_CODE=True)
#   [12] -> r+     : sinal + para o ajuste de raio (expandir)
#   [13] -> r-     : sinal − para o ajuste de raio (encolher)
#   [14..21] rad_u8 (8 bits, LSB-first):
#            magnitude do ajuste de raio
#            (pode vir em Gray code se USE_GRAY_CODE=True)
#
# Regras de “um lado só” (evitam conflito de sinais/direções):
#   - Eixo X: (x+, x-) → sx
#       (1,0) => +1  |  (0,1) => −1  |  (1,1) ou (0,0) => 0
#   - Eixo Y: (y+, y-) → sy
#       (1,0) => +1  |  (0,1) => −1  |  (1,1) ou (0,0) => 0
#   - Raio: (r+, r−) → sr
#       (1,0) => +1  |  (0,1) => −1  |  (1,1) ou (0,0) => 0
#
# Pesos:
#   • Bits 0..3 (direção do centro):
#       - Muito relevantes no início (init) e em white_scan (quando a imagem local é toda branca).
#       - Em “approach” e “border_seek”, a direção do centro é guiada por heurísticas (centroide/anel preto), então os bits 0..3
#         podem ter impacto menor nesses modos.
#   • Bits 4..11 (magnitude do movimento do centro):
#       - Usados no init (nn_initial_adjust) e no white_scan (define passo base/jitter). Em approach/border_seek, o passo vem
#         de heurísticas (distância × ganhos), logo essa magnitude tem pouco efeito direto.
#   • Bits 12..13 (sinal do raio):
#       - Importantes no init (primeiro delta_r).
#       - No loop principal, o SINAL do raio é decidido por testes de anel:
#           outer alto  -> expandir
#           inner baixo -> encolher
#         Ou seja, depois do início, estes bits quase não determinam o sinal sozinhos (o código usa a magnitude da rede + heurística
#         para o sinal).
#   • Bits 14..21 (magnitude do raio):
#       - Sempre relevantes: definem o “quanto” ajustar o raio quando a heurística manda expandir/encolher.
#       - Há caps contextuais (RAD_CAP_NEAR_PX/RAD_CAP_FAR_PX) para evitar overshoot, e a função compute_shrink_delta(...) pode
#         amplificar o encolhimento se o inner ring estiver muito longe do ideal (acelera correção quando o círculo está grande).
#
# Gray code (USE_GRAY_CODE=True):
#   - move_u8 e rad_u8 podem ser emitidos em Gray code (cada vizinho difere por 1 bit).
#   - Isso suaviza o “custo” de mutações binárias no AG: pequenos ajustes de magnitude tendem a mudar poucos bits,
#     deixando a busca mais estável.
#
# Resumo:
#   1) A REDE decide bits -> decode_actions(...) dá (sx, sy, move_step, sr, rad_step).
#   2) INIT:
#        - Move centro usando (sx, sy, move_step).
#        - Aplica um primeiro delta_r = sr * rad_mag (respeitando limites).
#   3) LOOP:
#        - Modos de deslocamento do centro (approach/border/white) são guiados por heurísticas;
#          os bits 0..3 e 4..11 ajudam mais quando a cena local é “branca” ou no começo.
#        - O ajuste de RAIO:
#            * sinal: heurísticas de anel (outer/inner) decidem expandir/encolher;
#            * magnitude: vem dos bits 14..21 (com caps e, para encolher, ganho adaptativo).
# ============================================================
ACTION_BITS = 22

def split_outputs(out):
    """
    Separadas as saídas em duas partes:
      - primeiros 22 índices: bits de ação (decidimos deslocamento e passo de raio);
      - o restante (se houver) pode carregar (cx, cy, r) normalizados para “init head”.
    """
    if out.shape[0] <= ACTION_BITS:
        return out, None
    return out[:ACTION_BITS], out[ACTION_BITS:]

# ============================================================
# 6) Decodificação de 22 bits (com Gray code)
# ============================================================
def _bits_to_uint8_lsb(bits8):
    """Converte 8 bits LSB-first (vetor de {0,1}) para inteiro [0..255]."""
    v = 0
    for i, b in enumerate(bits8):
        v |= (int(b) << i)
    return v & 0xFF

def _gray_to_binary_u8(g):
    """
    Converte de Gray code para binário clássico. Vantagem do Gray: vizinhos mudam
    por 1 bit, aumentando robustez a erros pequenos no AG.
    """
    g = int(g) & 0xFF
    b = g
    shift = 1
    while shift < 8:
        b ^= (b >> shift); shift <<= 1
    return b & 0xFF

def _decode_u8(bits8, use_gray=True):
    """Decodifica 8 bits em [0..255], usando Gray code se use_gray=True."""
    raw = _bits_to_uint8_lsb(bits8)
    return _gray_to_binary_u8(raw) if use_gray else raw

def _smooth_frac(u8, gamma=1.0):
    """
    Converte inteiro [0..255] em fração [0..1], com compressão/expansão via gamma.
    gamma<1 abre a escala em valores baixos (passos maiores cedo); gamma>1 suaviza.
    """
    x = max(0.0, min(255.0, float(u8))) / 255.0
    return x**gamma if gamma != 1.0 else x

def decode_actions(out_vec, r_curr):
    """
    Decodifica o vetor de saída da RNA (22 bits) em:
      - (sx, sy): sinais discretos {-1, 0, +1} de deslocamento x/y (4 bits: x+, x-, y+, y-).
      - move_step: magnitude do passo de movimento do centro, escalado por r_curr.
      - (sr): sinal do raio {-1, 0, +1} (2 bits: r+, r-).
      - rad_step: magnitude do passo de raio, escalado por r_curr.
    """
    bits_all = (out_vec > 0.5).astype(np.uint8)
    bits = bits_all[:ACTION_BITS]
    bx_pos, bx_neg, by_pos, by_neg = map(int, bits[0:4])  # 4 bits: eixos
    sx = 1 if (bx_pos and not bx_neg) else (-1 if (bx_neg and not bx_pos) else 0)
    sy = 1 if (by_pos and not by_neg) else (-1 if (by_neg and not by_pos) else 0)
    # 8 bits para magnitude do movimento (Gray code -> fração -> escala por r)
    k_move_u8 = _decode_u8(bits[4:12], use_gray=USE_GRAY_CODE)
    move_frac = _smooth_frac(k_move_u8, gamma=MOVE_GAMMA)
    move_step = float(move_frac) * float(max(1, r_curr))
    # 2 bits para sinal de raio
    br_up, br_down = int(bits[12]), int(bits[13])
    sr = 1 if (br_up and not br_down) else (-1 if (br_down and not br_up) else 0)
    # 8 bits para magnitude do raio
    k_rad_u8 = _decode_u8(bits[14:22], use_gray=USE_GRAY_CODE)
    rad_frac = _smooth_frac(k_rad_u8, gamma=RAD_GAMMA)
    rad_step = float(rad_frac) * float(max(1, r_curr))
    return sx, sy, move_step, sr, rad_step, bits

# ============================================================
# 7) Estado / inicialização / limites geométricos
# ============================================================
def build_input_vec(img_small_bin, cx, cy, r):
    """
    Vetor de entrada da RNA = [imagem 28x28 achatada (0/1), cx/IMG_SIZE, cy/IMG_SIZE, r/R_NORM].
    As 3 últimas features dão contexto geométrico do estado atual.
    """
    state = np.array([cx/IMG_SIZE, cy/IMG_SIZE, r/float(R_NORM)], dtype=np.float32)
    return np.concatenate([img_small_bin.flatten(), state], axis=0).astype(np.float32)

def r_fit_for_center(size, cx, cy):
    """Maior raio que cabe totalmente dentro da imagem com centro (cx, cy)."""
    return int(max(0, min(cx, cy, size-1-cx, size-1-cy)))

def initial_center_fit_all(size):
    """Centro no meio da imagem, raio máximo que cabe nesse centro."""
    cx = size // 2; cy = size // 2
    r  = r_fit_for_center(size, cx, cy)
    return cx, cy, r

def clamp_center_with_radius(size, cx, cy, r):
    """Clampa (cx, cy) para garantir que o círculo com raio r caiba completamente."""
    cx = int(np.clip(cx, r, size-1-r))
    cy = int(np.clip(cy, r, size-1-r))
    return cx, cy

def clamp_center_partial(size, cx, cy):
    """
    Versão para círculos parciais: apenas clampa (cx, cy) nos limites do canvas,
    permitindo que parte do círculo fique fora.
    """
    cx = int(np.clip(cx, 0, size - 1))
    cy = int(np.clip(cy, 0, size - 1))
    return cx, cy

def enforce_bounds_partial(size, cx, cy, r):
    """Garante (cx, cy) no canvas e raio dentro de [1, R_EXT_MAX], modo parcial."""
    cx, cy = clamp_center_partial(size, cx, cy)
    r = int(max(1, min(r, R_EXT_MAX)))
    return cx, cy, r

def apply_radius_partial(size, cx, cy, r, delta_r):
    """Aplica variação de raio permitindo círculos parciais."""
    r_new = max(1, r + int(delta_r))
    return enforce_bounds_partial(size, cx, cy, r_new)

def apply_radius_recenter_partial(size, cx, cy, r, delta_r):
    """
    No modo parcial, NÃO recuamos para dentro antes de crescer (mantemos centro).
    Isso facilita “encostar” em bordas próximas mesmo perto da margem.
    """
    return apply_radius_partial(size, cx, cy, r, delta_r)

def enforce_bounds(size, cx, cy, r):
    """Modo completo: raio não pode extrapolar a imagem, reclampa centro e raio."""
    rmax = r_fit_for_center(size, cx, cy)
    r = int(max(1, min(r, rmax)))
    cx, cy = clamp_center_with_radius(size, cx, cy, r)
    return cx, cy, r

def initial_center_random(size, rng, r_min_px=INIT_RANDOM_R_MIN_PX, r_max_frac=INIT_RANDOM_R_MAX_FRAC):
    """Inicialização aleatória plausível (raio e centro válidos)."""
    r_max_px = int(max(1, min(int(r_max_frac * size), size // 2)))
    r = int(rng.integers(low=max(1, r_min_px), high=r_max_px + 1))
    cx = int(rng.integers(low=r, high=size - r))
    cy = int(rng.integers(low=r, high=size - r))
    return cx, cy, r

def nn_radius_delta(sr, rad_step):
    """
    Constrói delta de raio inteiro a partir do sinal sr∈{-1,0,1} e da magnitude
    contínua sugerida (rad_step). Garante piso RAD_STEP_FLOOR_PX.
    """
    step = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step))))
    return int(sr) * step

def apply_radius(size, cx, cy, r, delta_r):
    """Aplica variação de raio e reimpõe limites rígidos (círculo completo)."""
    r_new = max(1, r + int(delta_r))
    cx, cy, r_new = enforce_bounds(size, cx, cy, r_new)
    return cx, cy, r_new

def apply_radius_recenter(size, cx, cy, r, delta_r):
    """
    Variante recenter: para crescer, recua o centro se necessário para caber.
    Útil quando não permitimos círculos parciais.
    """
    desired = max(1, r + int(delta_r))
    cx = int(np.clip(cx, desired, size - 1 - desired))
    cy = int(np.clip(cy, desired, size - 1 - desired))
    r_new = desired
    cx, cy, r_new = enforce_bounds(size, cx, cy, r_new)
    return cx, cy, r_new

def nn_initial_adjust(nn, img_small_bin, cx, cy, r, size=IMG_SIZE):
    """
    Um ou mais passos de chute inicial guiado pela RNA:
    - Move o centro conforme bits de direção + magnitude;
    - Ajusta o raio conforme sinal+magnitude.
    """
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
        cx, cy = clamp_center_with_radius(size, cx, cy, r)
        delta_r = nn_radius_delta(sr, rad_step_nn)
        cx, cy, r = apply_radius(size, cx, cy, r, delta_r)
        cx, cy, r = enforce_bounds(size, cx, cy, r)

    return cx, cy, r

def nn_init_head_propose(nn, img_small_bin, size, r_min_px=INIT_RANDOM_R_MIN_PX):
    """
    Quando USE_INIT_HEAD=True, a RNA tem 3 saídas extras (u_cx, u_cy, u_r) em [0,1]
    que sugerem diretamente uma proposta inicial (cx, cy, r).
    """
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
    """
    Escolhe estado inicial conforme política:
    - 'center_fit': centro no meio, raio máximo que cabe;
    - 'random_only': aleatório válido;
    - 'nn_only': começa do centro e aplica INIT_NN_STEPS passos guiados;
    - 'random_then_nn': aleatório e aplica INIT_NN_STEPS passos guiados;
    - 'nn_head_only' / 'random_then_nn_head': usam init head (se habilitado).
    """
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
        return enforce_bounds(size, cx, cy, r)

# ============================================================
# 8) Auxiliares heurísticos (sinal do raio, centróide, etc.)
# ============================================================
def compute_shrink_delta(rad_step_nn, inner_b, near):
    """
    Calcula delta negativo (encolher) baseado na magnitude sugerida pela RNA e
    no “déficit” do anel interno (quanto menor inner_b, mais evidente que estamos
    grandes demais). Aplica ganhos adaptativos e tetos por proximidade.
    """
    # 1) Magnitude base sugerida pela RNA (mantemos o piso)
    base = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))

    # 2) Déficit: quão abaixo do limiar EPS_INNER_SHRINK está o anel interno?
    deficit = 0.0
    if EPS_INNER_SHRINK > 1e-9:
        deficit = max(0.0, (EPS_INNER_SHRINK - inner_b) / float(EPS_INNER_SHRINK))
        deficit = min(deficit, 1.0)

    # 3) Ganho adaptativo dentro do intervalo [SHRINK_GAIN_MIN, SHRINK_GAIN_MAX]
    gain = SHRINK_GAIN_MIN + (SHRINK_GAIN_MAX - SHRINK_GAIN_MIN) * deficit

    # 4) Aplica teto (cap) diferente se está “perto” ou “longe” da borda
    cap = SHRINK_CAP_NEAR_PX if near else SHRINK_CAP_FAR_PX
    mag = min(int(round(gain * base)), cap)

    return -mag  # negativo = encolher

def any_black_interior(img255, cx, cy, r):
    """Retorna True se houver pelo menos um pixel preto dentro do círculo."""
    mask = _circle_mask(img255.shape[0], cx, cy, r)
    if not np.any(mask): return False
    return np.any(img255[mask] <= BLACK_THR)

def border_black_direction(img255, cx, cy, r, cos_tab, sin_tab):
    """
    Se há pixels pretos na borda do círculo, computa média vetorial apontando
    para onde o preto concentra. Isso dá uma direção útil para buscar a borda.
    """
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
    """
    Critério simples de círculo bom:
      - anel interno tem que estar bem preto (>=th_inner),
      - anel externo tem que estar bem claro (<=th_outer).
    """
    inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
    outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)
    return (inner_b >= th_inner) and (outer_b <= th_outer), inner_b, outer_b

def centroid_black_interior(img255, cx, cy, r):
    """
    Centróide (média) dos pixels pretos dentro do círculo.
    Ajuda a puxar o centro para o miolo da região preta.
    """
    size = img255.shape[0]
    mask = _circle_mask(size, cx, cy, r)
    ys, xs = np.where(mask & (img255 <= BLACK_THR))
    if xs.size == 0: return None
    mx = int(np.clip(int(np.rint(xs.mean())), 0, size - 1))
    my = int(np.clip(int(np.rint(ys.mean())), 0, size - 1))
    return (mx, my)

def circle_perfect(img255, cx, cy, r, cos_tab, sin_tab,
                   inner_req=PERFECT_INNER_FRAC, outer_req=PERFECT_OUTER_FRAC):
    """
    Critério de perfeito para encerrar cedo quando anéis grosso/fino batem
    exatamente: interior quase todo preto e exterior quase todo branco.
    """
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
# 9) Controlador (modo sem traço/GIF — apenas otimiza e devolve melhor)
# ============================================================
def run_controller(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, probe_r_list,
                   return_initial=False):
    """
    Executa laço de controle por “steps” iterações:
      - cada passo: RNA → ação (mover centro e ajustar raio),
      - heurísticas anti-stall (varredura branca, superjump),
      - heurísticas de raio (expandir/encolher com tetos),
      - atualiza “melhor visto” via loss,
      - paradas: perfect match, paciência, etc.
    Retorna (best_loss, cx, cy, r) e, opcionalmente, o estado inicial avaliado.
    """
    size = img255.shape[0]
    cx, cy, r = choose_initial_state(nn, img255, img_small_bin, size, rng,
                                     policy=INIT_POLICY, use_init_head=USE_INIT_HEAD)

    initial_loss = metrics_loss_fn(cx, cy, r)
    initial_state = (cx, cy, r, initial_loss)

    best = (initial_loss, cx, cy, r)
    no_improve = 0

    # Estados auxiliares do movimento
    vx = 0.0; vy = 0.0
    scan_dirs = [(1,0),(0,1),(-1,0),(0,-1)]  # padrão de varredura (direções ortogonais)
    scan_k = 0
    stuck_white = 0
    white_streak = 0
    prev_mode = "init"
    last_dist = None

    t = 0
    max_steps = steps
    max_steps_cap = steps + EXTRA_STEPS_CAP

    while t < max_steps:
        # 1) RNA decide a ação com base no estado atual (imagem binária + [cx, cy, r])
        x_in = build_input_vec(img_small_bin, cx, cy, r)
        out  = nn.forward(x_in)
        act, _ = split_outputs(out)
        sx, sy, move_step_nn, sr, rad_step_nn, _bits = decode_actions(act, r)

        # 2) Detecta sinal: há preto no interior? na borda?
        interior_black = any_black_interior(img255, cx, cy, r)
        border_dir = border_black_direction(img255, cx, cy, r, cos_tab, sin_tab)
        border_has_black = (border_dir is not None)
        all_white = (not interior_black) and (not border_has_black)

        # 3) Decide movimento do centro (dx, dy) conforme o modo:
        if all_white:
            # Varredura no branco: segue padrão + jitter e cresce passo com streak
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
            # super jump periódico para “sair do nada”
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
            # Seek para a borda: move na direção média onde detectamos preto na borda
            white_streak = 0
            step = int(max(1, min(BORDER_STEP_MAX_PX, int(r))))
            ux, uy = border_dir
            raw_dx = step * ux; raw_dy = step * uy
            mode = "border_seek"
            last_dist = None

        else:
            # Há interior preto: aproxima centróide (puxa centro pro miolo)
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

        # Amortece momento se entramos num modo fino
        if mode in ("approach", "border_seek") and mode != prev_mode:
            vx *= MOMENTUM_DAMP_ON_MODE_CHANGE
            vy *= MOMENTUM_DAMP_ON_MODE_CHANGE

        # Momentum + clamp do centro
        if abs(raw_dx) <= DEADZONE_PX: raw_dx = 0.0
        if abs(raw_dy) <= DEADZONE_PX: raw_dy = 0.0
        vx = MOMENTUM_BETA * vx + (1.0 - MOMENTUM_BETA) * raw_dx
        vy = MOMENTUM_BETA * vy + (1.0 - MOMENTUM_BETA) * raw_dy
        dx = int(round(vx)); dy = int(round(vy))
        new_cx = cx + dx; new_cy = cy + dy

        if ALLOW_PARTIAL_CIRCLE:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
        else:
            new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)

        # Se “travou” no branco, dá um salto em direção ao centro do canvas
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
                        new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        # ---------------- Crescimento de raio em “branco” ----------------
        white_growth_done = False
        if all_white:
            grow_ready = (white_streak >= WHITE_RADIUS_GROW_AFTER)
            if grow_ready:
                periodic_ok = ((white_streak - WHITE_RADIUS_GROW_AFTER) % max(1, WHITE_RADIUS_GROW_EVERY) == 0)
                bit_ok = True  # aqui magnitude é a da RNA, o “sinal” é dado pelo ramo
                if periodic_ok and bit_ok:
                    rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                    grow_px = int(min(rad_mag, WHITE_RADIUS_GROW_MAX_PX))
                    if grow_px > 0:
                        if ALLOW_PARTIAL_CIRCLE:
                            cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
                        else:
                            cx, cy, r = apply_radius_recenter(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds(size, cx, cy, r)
                        if WHITE_RADIUS_RESET_STREAK:
                            white_streak = 0
                        white_growth_done = True
        # ------------------------------------------------------------------------

        # 4) Heurística de sinal do raio com caps dependentes da distância
        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)

        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if outer_b > EPS_OUTER_EXPAND:
                # Expandir quando há preto fora (estamos cortando borda)
                rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                delta_r = +min(rad_mag, cap_px)  # só magnitude vem da RNA, sinal é heurístico
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_recenter(size, cx, cy, r, delta_r)
            elif interior_black and (inner_b < EPS_INNER_SHRINK):
                # Encolher quando anel interno está “claro demais”
                delta_r = compute_shrink_delta(rad_step_nn, inner_b, near)
                if delta_r < 0:
                    delta_r = -min(abs(delta_r), cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius(size, cx, cy, r, delta_r)

        # 5) Reforça limites
        if ALLOW_PARTIAL_CIRCLE:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
        else:
            cx, cy, r = enforce_bounds(size, cx, cy, r)

        # 6) Parada por “perfeito”
        if STOP_ON_PERFECT:
            is_perfect, _, _ = circle_perfect(img255, cx, cy, r, cos_tab, sin_tab)
            if is_perfect:
                l = metrics_loss_fn(cx, cy, r)
                if (l + IMPROVE_EPS) < best[0]:
                    best = (l, cx, cy, r)
                break

        # 7) Avalia e atualiza “melhor”
        l = metrics_loss_fn(cx, cy, r)
        if (l + IMPROVE_EPS) < best[0]:
            best = (l, cx, cy, r)
            if t >= WARMUP_STEPS: no_improve = 0
        else:
            if t >= WARMUP_STEPS:
                no_improve += 1
                if EARLY_STOP and (no_improve >= PATIENCE_STEPS): break

        # 8) Estende orçamento de passos quando há sinal
        if AUTO_EXTEND_STEPS and (max_steps < max_steps_cap):
            if (outer_b > EPS_OUTER_EXPAND) or (interior_black and inner_b < EPS_INNER_SHRINK) or (mode in ("approach",)):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_HAS_SIGNAL)
            elif mode in ("border_seek",):
                max_steps = min(max_steps_cap, steps + EXTRA_STEPS_BORDER)

        prev_mode = mode
        t += 1

    # 9) Snap refine final por IoU de máscara (pequena busca local para encaixar)
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
# 10) Controlador com tracing (gera GIF, persegue IoU, etc.)
# ============================================================
def run_controller_trace(nn, img255, img_small_bin, steps, cos_tab, sin_tab, metrics_loss_fn, gt_tuple=None):
    """
    Versão detalhada com rastreamento de cada passo:
    - Loga (t, cx, cy, r, loss, modo);
    - Pode perseguir IoU com GT;
    - Ao final, aplica snap refine por IoU de máscara.
    Retorna (trace, best_dict).
    """
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

    # Histórico para detectar ciclos ping-pong (2 estados alternados)
    prev1 = None
    prev2 = None

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

        if ALLOW_PARTIAL_CIRCLE:
            new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
        else:
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
                    if ALLOW_PARTIAL_CIRCLE:
                        new_cx, new_cy = clamp_center_partial(size, new_cx, new_cy)
                    else:
                        new_cx, new_cy = clamp_center_with_radius(size, new_cx, new_cy, r)
                stuck_white = 0
        else:
            stuck_white = 0

        cx, cy = new_cx, new_cy

        # ---- Decisão de raio (mesma lógica do modo não-trace, mas logando “mode_r”) ----
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
                            cx, cy, r = apply_radius_recenter(size, cx, cy, r, +grow_px)
                            cx, cy, r = enforce_bounds(size, cx, cy, r)
                        white_growth_done = True
                        mode_r = "white_grow"

        inner_b = _ring_fraction_vec(img255, cx, cy, r, delta=-1, cos_tab=cos_tab, sin_tab=sin_tab)
        outer_b = _ring_fraction_vec(img255, cx, cy, r, delta=+1, cos_tab=cos_tab, sin_tab=sin_tab)

        dist_for_cap = last_dist if (last_dist is not None) else max(1, r)
        near = (dist_for_cap < (APPROACH_NEAR_FRAC_OF_R * max(1, r)))
        cap_px = RAD_CAP_NEAR_PX if near else RAD_CAP_FAR_PX

        if not white_growth_done:
            if outer_b > EPS_OUTER_EXPAND:
                rad_mag = max(RAD_STEP_FLOOR_PX, int(round(abs(rad_step_nn))))
                delta_r = +min(rad_mag, cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_recenter_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius_recenter(size, cx, cy, r, delta_r)
                mode_r = "expand_nn"
            elif interior_black and (inner_b < EPS_INNER_SHRINK):
                delta_r = compute_shrink_delta(rad_step_nn, inner_b, near)
                if delta_r < 0:
                    delta_r = -min(abs(delta_r), cap_px)
                if ALLOW_PARTIAL_CIRCLE:
                    cx, cy, r = apply_radius_partial(size, cx, cy, r, delta_r)
                else:
                    cx, cy, r = apply_radius(size, cx, cy, r, delta_r)

        if ALLOW_PARTIAL_CIRCLE:
            cx, cy, r = enforce_bounds_partial(size, cx, cy, r)
        else:
            cx, cy, r = enforce_bounds(size, cx, cy, r)

        # (outras paradas como mask_iou_stop / ping-pong / perfect / IoU com GT
        #  podem ser inseridas aqui se precisarmos)

        l = metrics_loss_fn(cx, cy, r)
        trace.append({"t": t, "cx": int(cx), "cy": int(cy), "r": int(r),
                      "loss": float(l), "mode": mode if mode_r is None else mode_r})

        # Atualiza histórico para ping-pong (mantido como referência)
        prev2 = prev1
        prev1 = (cx, cy, r, (mode_r if mode_r is not None else mode))

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

    # Escolhe melhor frame do trace
    good = [p for p in trace if not (isinstance(p["loss"], float) and math.isnan(p["loss"]))]
    best_idx = int(np.argmin([p["loss"] for p in good])) if good else 0
    best = good[best_idx] if good else trace[-1]

    # Snap refine final (melhora local via IoU máscara)
    (snap_cx, snap_cy, snap_r), _ = snap_refine_mask_iou(
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
# 11) Desenho e geração de GIFs (visualização dos passos)
# ============================================================
def _draw_frame(img255, p, gt=None, scale=GIF_SCALE):
    """
    Desenha um frame RGB com a imagem original e os círculos:
      - predição (vermelho),
      - GT (verde), se fornecido e SHOW_GT_IN_GIF=True,
      - HUD com t, loss, (cx, cy, r) e modo.
    """
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

    if 'PRED_ON_TOP' in globals() and PRED_ON_TOP:
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
                   fill=HUD_TEXT_COLOR + ((255,) if len(HUD_TEXT_COLOR) == 3 else ()),
                   stroke_width=sw,
                   stroke_fill=HUD_STROKE_COLOR + ((255,) if len(HUD_STROKE_COLOR) == 3 else ()))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    else:
        draw.text((x0, y0), hud_text,
                  font=font,
                  fill=HUD_TEXT_COLOR,
                  stroke_width=sw,
                  stroke_fill=HUD_STROKE_COLOR)

    return base

def save_gif_for_trace(img255, trace, gt_tuple, out_path):
    """
    Gera um GIF do controle:
      - opcionalmente recorta no primeiro stop (TRIM_GIF_AT_STOP),
      - mantém frame de snap_refine após stop (KEEP_SNAP_AFTER_STOP).
    """
    if 'TRIM_GIF_AT_STOP' in globals() and TRIM_GIF_AT_STOP:
        trace = trim_trace_on_stop(trace, keep_snap=KEEP_SNAP_AFTER_STOP)

    gt_draw = gt_tuple if SHOW_GT_IN_GIF else None
    frames = [_draw_frame(img255, p, gt_draw) for p in trace]

    if len(frames) == 1:
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
# Pós-processamento do trace: recorte no primeiro stop + contagem steps
# ============================================================
# Ordem de prioridade dos motivos de parada:
STOP_MODES_ORDERED = ["iou_stop_equal", "iou_stop", "mask_iou_stop", "radius_ping_pong", "perfect_stop"]

def _first_stop_index(trace):
    """Encontra o primeiro índice no trace que contém um modo de parada reconhecido."""
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
    """Corta o trace no primeiro stop, opcionalmente inclui um frame de snap_refine."""
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
    """Conta quantos passos efetivos foram usados até o stop (ou até o fim do trace)."""
    idx, _ = _first_stop_index(trace)
    if idx is not None and isinstance(trace[idx].get("t"), int):
        return trace[idx]["t"] + 1
    t_last = None
    for p in trace:
        if isinstance(p.get("t"), int):
            t_last = p["t"]
    return (t_last + 1) if t_last is not None else None

# ============================================================
# 12) Configuração da rede / persistência (checkpoint)
# ============================================================
input_size   = IN_SIZE * IN_SIZE + 3
hidden_sizes = [6]  # Facilita o AG explorar rapidamente
output_size  = ACTION_BITS + (3 if USE_INIT_HEAD else 0)
nn = NeuralNetwork(input_size, hidden_sizes, output_size, use_init_head=USE_INIT_HEAD)

CHECKPOINT_TAG   = f"v12_overshoot_tamed_iouchase_borderaware_initpolicy_{INIT_POLICY}_inithead_{int(USE_INIT_HEAD)}"
CHECKPOINT_DIR   = "checkpoints"
CHECKPOINT_PATH  = os.path.join(CHECKPOINT_DIR, f"ckpt_single_ga_controller_{CHECKPOINT_TAG}.npz")

def _arch_signature(num_weights, num_bits):
    """Assinatura do arranjo de pesos/bits para validar compatibilidade ao carregar checkpoint."""
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
    """Salva pesos da RNA + população do AG + metadados (compatibilidade)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    weights = nn.get_weights().astype(np.float32)
    arch = _arch_signature(num_weights, num_bits)
    meta_json = json.dumps(arch)
    np.savez_compressed(path, weights=weights, pop=popx.astype(np.uint8), meta=meta_json)

def load_checkpoint(expected_num_weights, expected_num_bits, path=CHECKPOINT_PATH):
    """
    Carrega checkpoint se compatível com a arquitetura/assinatura atual.
    Retorna (weights, pop) ou (None, None) se incompatível/ausente.
    """
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
# 13) Carrega anotações (dataset)
# ============================================================
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]
annotations = [rec for rec in records if rec.get("split") == "multi"]

# ============================================================
# 14) Loop do AG + execução final (com GIFs e logs)
# ============================================================

initial_population = None
_num_weights = nn.get_weights().size                 # nº de pesos reais da RNA
_num_bits    = int(_num_weights * 16)                # cromossomo: 16 bits por peso (int16 -> float/1000)

# Tenta carregar checkpoint (retoma de onde parou)
ckpt_weights, ckpt_pop = load_checkpoint(_num_weights, _num_bits, CHECKPOINT_PATH)
if ckpt_weights is not None:
    nn.set_weights(ckpt_weights.astype(np.float32))
    print(f"[ckpt] Weights loaded: {CHECKPOINT_PATH} (num_weights={_num_weights})")
if ckpt_pop is not None:
    initial_population = ckpt_pop.astype(np.uint8)
    print(f"[ckpt] GA population loaded: {initial_population.shape}")

# Estatísticas gerais do dataset
sum_iou = 0.0
cnt = 0
cnt_iou_good = 0
cnt_stuck = 0

# Arquivo de log (JSONL) + contador de GIFs
RUNS_DIR and os.path.exists(RUNS_DIR)
logf = open(RUN_JSONL_PATH, 'w', encoding='utf-8')
gif_count = 0

# Definições para a busca sequencial
MAX_BALLS = 5          # Limite máximo de bolas a buscar por imagem
STOP_LOSS_THR = 1.0    # Parada heurística: se a loss for muito alta, para a busca
MIN_BLACK_PIXELS = 10  # Mínimo de pixels pretos para continuar buscando

for ann in annotations:
    file_rel = ann["file"]
    file_path = os.path.join(DATA_ROOT, file_rel)
    img_small = load_image_small_bin(file_path, out_size=IN_SIZE)
    img_full  = load_image_full_gray(file_path)

    # Listas de “probe” para fases grossa e fina
    probe_list_coarse = _make_probe_list(PROBE_R_COARSE, img_full.shape[0])
    probe_list_fine   = _make_probe_list(PROBE_R_FINE,   img_full.shape[0])

    # Caches de loss (evita recomputar nos mesmos estados inteiros)
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

    # Ground truth (um único círculo por imagem nesse split, usado apenas para log)
    circle = ann["circles"][0]
    x_real, y_real, r_real = int(circle["cx"]), int(circle["cy"]), int(circle["r"])

    # Avalia “estado inicial” (sem dar passos) para calibrar steps finos
    base_loss, cx0, cy0, r0, initial_state = run_controller(
        nn, img_full, img_small,
        steps=0,
        cos_tab=COS_COARSE, sin_tab=SIN_COARSE,
        metrics_loss_fn=metrics_loss_coarse,
        probe_r_list=probe_list_coarse,
        return_initial=True
    )
    cx_init, cy_init, r_init, loss_init_true = initial_state

    # Função de fitness da fase grossa (o AG avalia vários cromossomos aqui)
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

    # Configurações do AG (fase 1)
    popsize_stage1 = initial_population.shape[0] if (initial_population is not None) else GA_POP_COARSE
    gaoptions1 = {
        "PopulationSize": popsize_stage1,
        "Generations": GA_GENS_COARSE,
        "InitialPopulation": initial_population,
        "MutationFcn": GA_MUT_COARSE,
        "EliteCount": ELITE_COUNT,
    }
    x_best, popx, fitvals = gago(fit_func_coarse, _num_bits, gaoptions1)

    # Ajuste adaptativo do nº de passos finos conforme “quão boa” estava a loss inicial
    if loss_init_true <= 0.6:
        steps_fine = max(10, CTRL_STEPS_FINE - 2)
    elif loss_init_true <= 1.2:
        steps_fine = CTRL_STEPS_FINE
    else:
        steps_fine = max(CTRL_STEPS_FINE, 16)

    # Fitness para fase fina (polimento)
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

    # AG (fase 2)
    popsize_stage2 = popx.shape[0]
    gaoptions2 = {
        "PopulationSize": popsize_stage2,
        "Generations": GA_GENS_POLISH,
        "InitialPopulation": popx,
        "MutationFcn": GA_MUT_POLISH,
        "EliteCount": ELITE_COUNT,
    }
    x_best, popx, fitvals = gago(fit_func_fine, _num_bits, gaoptions2)

    # Converte melhor cromossomo para pesos e fixa na RNA
    best_weights = bits2bytes(x_best, 'int16').astype(np.float32) / 1000.0
    nn.set_weights(best_weights)

    # Atualiza estado do AG para próxima imagem e salva checkpoint
    initial_population = popx
    save_checkpoint(nn, initial_population, _num_weights, _num_bits, CHECKPOINT_PATH)

    # ============================================================
    # 15) NOVO LOOP DE DETECÇÃO SEQUENCIAL DE MÚLTIPLAS BOLAS
    # ============================================================
    
    current_img_full = img_full.copy()
    balls_found = []
    
    print(f"\n--- Iniciando busca sequencial para {file_rel} ---")
    
    while check_for_black(current_img_full) and len(balls_found) < MAX_BALLS:
        
        # 1. Verifica se ainda há pixels pretos suficientes para justificar a busca
        if np.count_nonzero(current_img_full <= BLACK_THR) < MIN_BLACK_PIXELS:
            print("  [STOP] Pixels pretos insuficientes restantes.")
            break
            
        # 2. Prepara a entrada da RNA com a imagem mascarada (usando a nova função)
        current_img_small = array_to_image_small_bin(current_img_full, out_size=IN_SIZE)
        
        # 3. Executa o controlador (busca a melhor bola remanescente)
        trace, best = run_controller_trace(
            nn, current_img_full, current_img_small,
            steps=int(steps_fine * max(1, INFER_STEPS_MULT)),
            cos_tab=COS_FINE, sin_tab=SIN_FINE,
            metrics_loss_fn=metrics_loss_fine,
            gt_tuple=(x_real, y_real, r_real) # GT é mantido apenas para referência visual/log
        )
        
        final_loss = float(best["loss"])
        cx_pred, cy_pred, r_pred = int(best["cx"]), int(best["cy"]), int(best["r"])
        
        # 4. Critério de Aceitação/Parada
        if final_loss > STOP_LOSS_THR:
            print(f"  [STOP] Loss final ({final_loss:.4f}) acima do limite {STOP_LOSS_THR:.2f}. Assumindo ruído.")
            break
            
        # 5. Mascaramento e Armazenamento
        
        # Apaga a bola encontrada da imagem para o próximo loop
        current_img_full = mask_circle_in_image(current_img_full, cx_pred, cy_pred, r_pred)
        
        # Armazena a predição
        balls_found.append({"x": cx_pred, "y": cy_pred, "r": r_pred, "loss": final_loss})

        # Opcional: Geração de GIF para esta bola (se ativado e dentro do limite)
        if MAKE_GIFS and (GIF_LIMIT is None or gif_count < GIF_LIMIT):
            safe_name = file_rel.replace("/", "__")
            gif_path = os.path.join(GIFS_DIR, f"{os.path.splitext(safe_name)[0]}_ball_{len(balls_found)}_{RUN_ID}.gif")
            # Salva o trace apenas da busca por esta bola
            # Nota: O GT original (x_real, y_real, r_real) será desenhado no GIF
            save_gif_for_trace(img_full, trace, (x_real, y_real, r_real), gif_path) 
            gif_count += 1
            print(f"  [gif] salvo para Bola {len(balls_found)}: {gif_path}")
        
    # ============================================================
    # 16) LOGS E SUMÁRIO (ADAPTADO)
    # ============================================================
    
    cnt += 1 # Conta a imagem no dataset

    # Logs legíveis no console
    print(f"\nResultado Final para Imagem: {file_path}")
    print(f"GT (px) ref:    (x={x_real}, y={y_real}, r={r_real})")
    print(f"Bolas Encontradas: {len(balls_found)}")
    
    total_iou_img = 0.0
    
    for i, ball in enumerate(balls_found):
        # Para fins de demonstração, calculamos o IoU contra o PRIMEIRO GT (x_real, y_real, r_real)
        # Se o dataset tiver múltiplos GTs, esta comparação precisa de emparelhamento.
        iou_val = iou_circle(IMG_SIZE, (ball["x"], ball["y"], ball["r"]), (x_real, y_real, r_real))
        total_iou_img += iou_val
        
        # Métricas simplificadas para o log da bola individual
        fill = interior_fill_fraction(img_full, ball["x"], ball["y"], ball["r"])
        
        print(f"  -> Bola {i+1} (px): (x={ball['x']}, y={ball['y']}, r={ball['r']})")
        print(f"     Loss: {ball['loss']:.4f} | Fill: {fill:.4f} | IoU vs GT1: {iou_val:.4f}")

    if len(balls_found) > 0:
        sum_iou += (total_iou_img / len(balls_found)) # Média IoU desta imagem (simplificada)
        if total_iou_img > 0.5: cnt_iou_good += 1 # Condição simplificada de "boa" detecção
    
    # Linha JSONL detalhada para análise posterior (ADAPTADA)
    log_line = {
        "run_id": RUN_ID,
        "file": file_rel,
        "gt_ref": {"x": x_real, "y": y_real, "r": r_real},
        "init": {"x": cx_init, "y": cy_init, "r": r_init, "loss": float(loss_init_true)},
        "predicoes": balls_found,
        "total_bolas_encontradas": len(balls_found),
        "ctrl": {"coarse_steps": CTRL_STEPS_COARSE, "fine_steps": steps_fine,
                 "policy": "SEQUENCIAL_SEARCH",
                 "search_budget": SEARCH_BUDGET, "infer_steps_mult": INFER_STEPS_MULT},
        "time": time.time()
    }
    logf.write(json.dumps(log_line, ensure_ascii=False) + "\n")
    logf.flush()

# Resumo final do dataset (útil para acompanhar evolução ao longo de execuções)
if cnt > 0:
    # As variáveis cnt, sum_iou, cnt_iou_good, e cnt_stuck são acumuladas no loop principal
    # e agora refletem a Média Simplificada de IoU por imagem.
    mean_iou  = sum_iou / float(cnt)
    pct_good  = 100.0 * cnt_iou_good / float(cnt)
    # Mantemos pct_stuck no sumário, mas a lógica de acumulação (cnt_stuck) não foi detalhada
    # no loop sequencial, portanto, será 0 ou imprecisa.
    pct_stuck = 100.0 * cnt_stuck / float(cnt) 
    
    print(f"\n[Resumo dataset] imagens={cnt} | IoU médio (simplificado)={mean_iou:.3f} | %Imagens c/ IoU>0.5={pct_good:.1f}% | %stuck≈2.0={pct_stuck:.1f}%")

    # Estrutura de Sumário JSONL Adaptada
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
            "GA_POP": GA_POP_COARSE, "GA_GENS_COARSE": GA_GENS_COARSE, "GA_GENS_POLISH": GA_GENS_POLISH,
            "GA_MUT_COARSE": GA_MUT_COARSE, "GA_MUT_POLISH": GA_MUT_POLISH,
            "MOMENTUM_BETA": MOMENTUM_BETA, "USE_GRAY_CODE": USE_GRAY_CODE,
            "POLICY": "SEQUENCIAL_SEARCH (white_scan | border_seek | approach)",
            "AUTO_EXTEND_STEPS": AUTO_EXTEND_STEPS,
            "EXTRA_STEPS_CAP_TRACE": EXTRA_STEPS_CAP_TRACE,
            "SEARCH_BUDGET": SEARCH_BUDGET, "INFER_STEPS_MULT": INFER_STEPS_MULT,
            "SEQUENTIAL_CONFIG": {
                "MAX_BALLS": MAX_BALLS,
                "STOP_LOSS_THR": STOP_LOSS_THR,
                "MIN_BLACK_PIXELS": MIN_BLACK_PIXELS
            }
        }
    }
    with open(RUN_JSONL_PATH, 'a', encoding='utf-8') as fsum:
        fsum.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(f"[Logs] JSONL salvo em: {RUN_JSONL_PATH}")
