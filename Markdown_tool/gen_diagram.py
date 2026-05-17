"""Generate PointNet++ architecture diagram as SVG using matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colours (full 6-digit hex for matplotlib 3.1)
ENC_BG   = '#E3F2FD'
ENC_EDGE = '#1565C0'
ENC_FONT = '#0D47A1'
DEC_BG   = '#E8F5E9'
DEC_EDGE = '#2E7D32'
DEC_FONT = '#1B5E20'
HEAD_BG  = '#F3E5F5'
HEAD_EDGE= '#7B1FA2'
HEAD_FONT= '#4A148C'
OUT_BG   = '#FFF3E0'
OUT_EDGE = '#E65100'
OUT_FONT = '#BF360C'
INP_BG   = '#F5F5F5'
INP_EDGE = '#616161'
INP_FONT = '#212121'
SKIP_CLR = '#FF6F00'
BOT_CLR  = '#1565C0'
LBL_CLR  = '#78909C'
GRY3     = '#333333'
GRY6     = '#666666'
GRY8     = '#888888'


def round_rect(x, y, w, h, fc, ec, lw=1.5, r=0.08):
    """Return a FancyBboxPatch (rounded rect)."""
    rp = r * min(w, h)
    return mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=mpatches.BoxStyle("Round", pad=rp/w, rounding_size=rp),
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)


def block(ax, x, y, w, h, text, bg, edge, fc, fs=8, align='center'):
    """Rounded rect + centred text."""
    ax.add_patch(round_rect(x, y, w, h, bg, edge))
    ax.text(x + w/2, y + h/2, text, fontsize=fs, fontfamily='monospace',
            color=fc, ha=align, va='center', zorder=4)


def label(ax, x, y, text, fs=8, colour=LBL_CLR, align='center'):
    ax.text(x, y, text, fontsize=fs, fontfamily='monospace',
            color=colour, ha=align, va='center', zorder=5)


def arrow(ax, x1, y1, x2, y2, colour=GRY3, lw=2, rad=0, style='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=colour, lw=lw,
                               linestyle=style,
                               connectionstyle=f'arc3,rad={rad}'),
                zorder=2)


def dashed_arrow(ax, x1, y1, x2, y2, colour, lw=1.8, rad=0):
    """Draw a dashed arrow using a Line2D + manual arrowhead."""
    import matplotlib.lines as mlines
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length < 0.01:
        return
    tx = dx / length
    ty = dy / length

    line = mlines.Line2D([x1, x2], [y1, y2], color=colour, lw=lw,
                         linestyle='dashed', zorder=2)
    ax.add_line(line)

    # Arrowhead at endpoint
    head_len = 0.12
    head_wid = 0.06
    ax.fill([x2, x2 - head_len*tx + head_wid*ty, x2 - head_len*tx - head_wid*ty],
            [y2, y2 - head_len*ty - head_wid*tx, y2 - head_len*ty + head_wid*tx],
            color=colour, zorder=2)


# ═══════════════════ Layout ═══════════════════
ENC_X = 1.2
DEC_X = 7.0
BLK_W = 3.0
BLK_H = 1.1
INP_H = 0.7
SML_H = 0.7

ROW1_B = 8.2   # Input / FP_1
ROW2_B = 6.5   # SA_1 / FP_2
ROW3_B = 4.8   # SA_2 / FP_3
ROW4_B = 3.1   # SA_3  (bottleneck row)

ENC_C = ENC_X + BLK_W/2
DEC_C = DEC_X + BLK_W/2
SKIP_X = (ENC_X + BLK_W + DEC_X) / 2  # mid-point for labels

# Subtitle row
label(ax, 7, 9.6, 'PointNet++ Encoder-Decoder Architecture for 3D Segmentation',
      13, GRY3, 'center')
label(ax, 7, 9.2,
      'Encoder: 3 × SA_MSG (down-sampling)  |  Decoder: 3 × FP (up-sampling + skip)  |  Head: Conv1d',
      8, GRY8, 'center')

# Direction hints
label(ax, ENC_C, ROW4_B - 0.5, '▼  Encoder (down-sampling)', 9, ENC_FONT)
label(ax, DEC_C, ROW4_B - 0.5, '▲  Decoder (up-sampling)', 9, DEC_FONT)

# Point-count labels
pl = {'fontsize': 8, 'color': LBL_CLR, 'fontfamily': 'monospace', 'ha': 'left', 'va': 'center'}
ax.text(0.2, ROW1_B + INP_H/2, '1024 pts', **pl)
ax.text(0.2, ROW2_B + BLK_H/2, '256 pts', **pl)
ax.text(0.2, ROW3_B + BLK_H/2, '64 pts', **pl)
ax.text(0.2, ROW4_B + BLK_H/2, '16 pts', **pl)

# ═══════════════════ Encoder ═══════════════════
block(ax, ENC_X, ROW1_B, BLK_W, INP_H,
      'Input\n[B, 3, 1024]', INP_BG, INP_EDGE, INP_FONT, 8.5)

block(ax, ENC_X, ROW2_B, BLK_W, BLK_H,
      'SA_MSG 1\nFPS 1024→256  MSG r=0.1/0.2\nMLP [16,32][32,64] + Attn\n→ [B, 96, 256]',
      ENC_BG, ENC_EDGE, ENC_FONT, 7.8)

block(ax, ENC_X, ROW3_B, BLK_W, BLK_H,
      'SA_MSG 2\nFPS 256→64  MSG r=0.2/0.4\nMLP [32,64][64,128] + Attn\n→ [B, 192, 64]',
      ENC_BG, ENC_EDGE, ENC_FONT, 7.8)

block(ax, ENC_X, ROW4_B, BLK_W, BLK_H,
      'SA_MSG 3\nFPS 64→16  MSG r=0.4/0.8\nMLP [64,128][128,256] + Attn\n→ [B, 384, 16]',
      ENC_BG, ENC_EDGE, ENC_FONT, 7.8)

# ═══════════════════ Decoder ═══════════════════
block(ax, DEC_X, ROW1_B, BLK_W, BLK_H,
      'FP Module 1\nInterp 256→1024 (dist-weighted)\n+ skip: raw xyz [3]\nMLP [64,64] → [B, 64, 1024]',
      DEC_BG, DEC_EDGE, DEC_FONT, 7.8)

block(ax, DEC_X, ROW2_B, BLK_W, BLK_H,
      'FP Module 2\nInterp 64→256 (dist-weighted)\n+ skip: SA_1 feat [96]\nMLP [128,64] → [B, 64, 256]',
      DEC_BG, DEC_EDGE, DEC_FONT, 7.8)

block(ax, DEC_X, ROW3_B, BLK_W, BLK_H,
      'FP Module 3\nInterp 16→64 (dist-weighted)\n+ skip: SA_2 feat [192]\nMLP [256,128] → [B, 128, 64]',
      DEC_BG, DEC_EDGE, DEC_FONT, 7.8)

# ═══════════════════ Head & Output ═══════════════════
HEAD_X = DEC_X + BLK_W + 0.5
OUT_X  = HEAD_X + 2.4 + 0.3

block(ax, HEAD_X, ROW1_B + 0.15, 2.4, SML_H,
      'Seg Head\nConv1d(64, C, 1)\nper-point cls',
      HEAD_BG, HEAD_EDGE, HEAD_FONT, 7.8)

block(ax, OUT_X, ROW1_B + 0.2, 2.2, SML_H,
      'Output\n[B, num_classes, 1024]\nper-point logits',
      OUT_BG, OUT_EDGE, OUT_FONT, 7.8)

# ═══════════════════ Main vertical arrows ═══════════════════
arrow(ax, ENC_C, ROW1_B + INP_H, ENC_C, ROW2_B, '#616161', 2)
arrow(ax, ENC_C, ROW2_B + BLK_H, ENC_C, ROW3_B, ENC_EDGE, 2)
arrow(ax, ENC_C, ROW3_B + BLK_H, ENC_C, ROW4_B, ENC_EDGE, 2)

# Bottleneck (SA_3 → FP_3, curve)
ax.annotate('', xy=(DEC_X, ROW3_B + BLK_H/2),
            xytext=(ENC_X + BLK_W, ROW4_B + BLK_H/2),
            arrowprops=dict(arrowstyle='->', color=BOT_CLR, lw=2.5,
                           connectionstyle='arc3,rad=-0.3'),
            zorder=2)
label(ax, (ENC_X + BLK_W + DEC_X) / 2, ROW4_B - 0.3, 'bottleneck (384-dim)',
      7.5, BOT_CLR)

# Decoder upward arrows
arrow(ax, DEC_C, ROW3_B, DEC_C, ROW2_B + BLK_H, DEC_EDGE, 2)     # FP_3 → FP_2
arrow(ax, DEC_C, ROW2_B, DEC_C, ROW1_B + BLK_H, DEC_EDGE, 2)     # FP_2 → FP_1

# FP_1 → Head
arrow(ax, DEC_X + BLK_W, ROW1_B + BLK_H/2, HEAD_X, ROW1_B + 0.15 + SML_H/2, DEC_EDGE, 2)

# Head → Output
arrow(ax, HEAD_X + 2.4, ROW1_B + 0.15 + SML_H/2,
      OUT_X, ROW1_B + 0.2 + SML_H/2, HEAD_EDGE, 2)

# ═══════════════════ Skip connections (dashed arrows) ═══════════════════
# SA_2 → FP_3
dashed_arrow(ax, ENC_X + BLK_W, ROW3_B + BLK_H/2,
             DEC_X, ROW3_B + BLK_H/2 + 0.05, SKIP_CLR)
label(ax, SKIP_X, ROW3_B + BLK_H/2 + 0.35, 'skip feat [B,192,64]', 7.2, SKIP_CLR)

# SA_1 → FP_2
dashed_arrow(ax, ENC_X + BLK_W, ROW2_B + BLK_H/2,
             DEC_X, ROW2_B + BLK_H/2, SKIP_CLR)
label(ax, SKIP_X, ROW2_B + BLK_H/2 + 0.35, 'skip feat [B,96,256]', 7.2, SKIP_CLR)

# Input → FP_1
dashed_arrow(ax, ENC_X + BLK_W, ROW1_B + INP_H/2,
             DEC_X, ROW1_B + BLK_H/2, SKIP_CLR)
label(ax, SKIP_X, ROW1_B + INP_H/2 + 0.4, 'skip xyz [B,3,1024]', 7.2, SKIP_CLR)

# ═══════════════════ Legend ═══════════════════
LX, LY = 0.5, 0.2
label(ax, LX, LY + 1.5, 'Legend:', 8, GRY6, 'left')

leg = [
    (ENC_BG, ENC_EDGE, 'Encoder (SA_MSG: FPS → MSG → MLP → Attention)'),
    (DEC_BG, DEC_EDGE, 'Decoder (FP: Interpolation + Skip + MLP)'),
    (HEAD_BG, HEAD_EDGE, 'Segmentation Head (Conv1d per-point)'),
]
for i, (bg, ec, txt) in enumerate(leg):
    y = LY + 1.1 - i * 0.3
    ax.add_patch(round_rect(LX, y, 0.3, 0.2, bg, ec, lw=1, r=0.02))
    label(ax, LX + 0.5, y + 0.1, txt, 7, GRY3, 'left')

# Skip legend line
import matplotlib.lines as mlines
ax.add_line(mlines.Line2D([LX, LX + 0.3], [LY + 0.45, LY + 0.45],
                           color=SKIP_CLR, linestyle='dashed', lw=1.5, zorder=2))
label(ax, LX + 0.5, LY + 0.45, 'Skip connection', 7, GRY3, 'left')

plt.savefig('/home/zjy/pointnetpp_architecture.svg', dpi=150,
            bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print("SVG saved OK")
