#!/usr/bin/env python3
"""v3.0 Post-Registration PDF generator.

Generates the post-registration report for:
  DOI: 10.5281/zenodo.24581149
  Date: 2026-11-27

Reads results from v30_postreg_results.json and v30_postreg_data.npz
(produced by run_v30_postreg.py).
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib import ticker
from scipy.stats import spearmanr

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, HRFlowable, PageBreak)

# ── Paths ──
_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(_DIR, 'v30_postreg_results.json')
NPZ_PATH  = os.path.join(_DIR, 'v30_postreg_data.npz')

with open(JSON_PATH) as f:
    DATA = json.load(f)
NPZ = np.load(NPZ_PATH)

results = DATA['results']
predictions = DATA['predictions']
overlap = DATA['overlap']
config = DATA['config']
summary = DATA['summary']

# ── Frozen module info ──
sys.path.insert(0, os.path.join(_DIR, 'frozen'))
import mtag_v24
BAND_Z  = np.array(sorted([d[0] for d in mtag_v24.DISKS]))
BAND_HW = np.array([d[2] for d in sorted(mtag_v24.DISKS, key=lambda d: d[0])])
_sensor_z = set(d[0] for d in mtag_v24._SENSOR_DISKS)
IS_SENSOR = np.array([z in _sensor_z for z in BAND_Z])
D_BAND = config['d_band_mm']

# ── Font ──
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
for name, path in [
    ('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    ('DejaVu-Italic', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'),
    ('DejaVu-BoldItalic', '/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf'),
]:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))

FONT = 'DejaVu'
FONTB = 'DejaVu-Bold'
FONTI = 'DejaVu-Italic'

# ── Colours ──
BLUE   = HexColor('#2c5f8a')
GREY   = HexColor('#888888')
LTGREY = HexColor('#dddddd')
GREEN  = HexColor('#1b7a3d')
RED    = HexColor('#c0392b')
AMBER  = HexColor('#e67e22')
WHITE  = HexColor('#ffffff')
LTYELLOW = HexColor('#fff8e1')

# ── Styles ──
sTitle = ParagraphStyle('Title', fontName=FONTB, fontSize=16, leading=20,
                         textColor=BLUE, alignment=TA_CENTER, spaceAfter=4*mm)
sSubTitle = ParagraphStyle('SubTitle', fontName=FONT, fontSize=10, leading=13,
                            textColor=GREY, alignment=TA_CENTER, spaceAfter=6*mm)
sH1 = ParagraphStyle('H1', fontName=FONTB, fontSize=13, leading=16,
                       textColor=BLUE, spaceBefore=8*mm, spaceAfter=3*mm)
sH2 = ParagraphStyle('H2', fontName=FONTB, fontSize=11, leading=14,
                       textColor=BLUE, spaceBefore=5*mm, spaceAfter=2*mm)
sBody = ParagraphStyle('Body', fontName=FONT, fontSize=9, leading=12,
                         alignment=TA_JUSTIFY, spaceAfter=2*mm)
sBullet = ParagraphStyle('Bullet', fontName=FONT, fontSize=9, leading=12,
                           leftIndent=10*mm, spaceAfter=1.5*mm, alignment=TA_JUSTIFY)
sCaption = ParagraphStyle('Caption', fontName=FONTI, fontSize=8, leading=10,
                            textColor=GREY, spaceAfter=3*mm, alignment=TA_JUSTIFY)
sFigLabel = ParagraphStyle('FigLabel', fontName=FONTB, fontSize=9, leading=11,
                             spaceBefore=2*mm, spaceAfter=1*mm)
sFooter = ParagraphStyle('Footer', fontName=FONTI, fontSize=7.5, leading=10,
                           textColor=GREY, alignment=TA_CENTER, spaceBefore=3*mm)

# Table styles
sTH  = ParagraphStyle('TH', fontName=FONTB, fontSize=8, leading=10, textColor=WHITE)
sTD  = ParagraphStyle('TD', fontName=FONT, fontSize=8, leading=10, alignment=TA_CENTER)
sTDL = ParagraphStyle('TDL', fontName=FONT, fontSize=8, leading=10, alignment=TA_LEFT)
sTDB = ParagraphStyle('TDB', fontName=FONTB, fontSize=8, leading=10, alignment=TA_CENTER)

# Shorthand
lam  = "λ"
dchi = "Δχ<super>2</super>"
dchiz = "Δχ<super>2</super><sub>z</sub>"
psub = "p<sub>conv</sub>"


def make_table(header, rows, col_widths):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), FONTB),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, LTGREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, HexColor('#f7f9fc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t


# ════════════════════════════════════════════════════════════
# Figure generation
# ════════════════════════════════════════════════════════════

C = dict(
    sensor='#3B82F6', service='#F59E0B', near='#60A5FA', inter='#EF4444',
    all_conv='#D1D5DB', untrim='#EF4444', trim='#10B981', ref_line='#9CA3AF',
    confirm='#10B981', refute='#EF4444', derived='#8B5CF6',
    bg='#FFFFFF', text='#1F2937', grid='#E5E7EB', accent='#6366F1',
    gold='#F59E0B', pink='#EC4899',
)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.facecolor': C['bg'], 'figure.facecolor': C['bg'],
    'axes.edgecolor': C['grid'], 'axes.grid': True,
    'grid.color': C['grid'], 'grid.alpha': 0.5, 'text.color': C['text'],
})


def fig1_dchi2z_distribution():
    """Fig 1: Δχ²_z distribution of removed vs surviving (seed 50)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    dchi2z_all = NPZ['dchi2z_all']
    near_all = NPZ['near_all']
    dchi2z_rem = NPZ['dchi2z_removed']
    near_rem = NPZ['near_removed']

    # Surviving mask: actual surviving near-band vertices (excluding removed)
    surv_mask = NPZ['surviving_mask']
    surv_near = near_all & surv_mask

    dchi2z_surv = dchi2z_all[surv_near]  # actual surviving near-band only
    dchi2z_rem_nb = dchi2z_rem[near_rem]

    # Panel a: histogram — extend x-axis to cover removed tail
    ax = axes[0]
    x_max = max(40, np.percentile(dchi2z_rem_nb, 99) if len(dchi2z_rem_nb) > 0 else 40)
    bins = np.linspace(0, x_max, 80)
    ax.hist(dchi2z_surv, bins=bins, color=C['all_conv'], alpha=0.6, label='Surviving near-band', density=True, zorder=1)
    ax.hist(dchi2z_rem_nb, bins=bins, color=C['untrim'], alpha=0.75, label='Removed near-band', density=True, zorder=2)
    ax.axvline(4.0, color=C['gold'], ls='--', lw=2, label='2σ boundary (Δχ²_z = 4)')
    ax.set_xlabel('Δχ²_z', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Δχ²_z Distribution (seed 50)', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)

    seed0 = results[0]
    ax.text(0.98, 0.95, f"Mean(rem) = {seed0['a1_mean_rem']:.2f}\n"
            f"Mean(surv) = {seed0['a1_mean_surv']:.2f}\n"
            f"Ratio = {seed0['a1_ratio']:.2f}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor=C['untrim'], alpha=0.9))

    # Panel b: ratio across seeds
    ax = axes[1]
    a1_arr = [r['a1_ratio'] for r in results if r['a1_ratio'] is not None]
    seeds_x = [r['seed_i'] for r in results if r['a1_ratio'] is not None]
    ax.scatter(seeds_x, a1_arr, c=C['untrim'], s=40, zorder=3, alpha=0.8, edgecolors='white', lw=0.5)
    ax.axhline(2.0, color=C['gold'], ls='--', lw=2, label='A1 threshold (2.0)')
    mean_a1 = np.mean(a1_arr)
    ax.axhline(mean_a1, color=C['accent'], ls='-', lw=2, label=f'Mean = {mean_a1:.2f}')
    ax.set_xlabel('Seed index i', fontsize=11)
    ax.set_ylabel('Mean Δχ²_z ratio (removed / surviving)', fontsize=11)
    ax.set_title('A1: Ratio Across 50 Seeds', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle('Investigation A — Δχ²_z Distribution Comparison',
                 fontsize=14, fontweight='bold', color=C['accent'], y=1.02)
    fig.tight_layout()
    p = os.path.join(_DIR, 'fig1_inv_a_dchi2z.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return p


def fig2_lambda_correlation():
    """Fig 2: λ–Δχ²_z correlation and B3 AUC."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: scatter |λ| vs Δχ²_z (seed 50)
    ax = axes[0]
    lam_all = NPZ['lam_all_base']
    dchi2z_all = NPZ['dchi2z_all']
    near_all = NPZ['near_all']

    lam_nb = np.abs(lam_all[near_all])
    dz_nb = dchi2z_all[near_all]

    # Subsample for scatter
    rng = np.random.default_rng(42)
    n_plot = min(5000, len(lam_nb))
    idx = rng.choice(len(lam_nb), n_plot, replace=False)
    ax.scatter(dz_nb[idx], lam_nb[idx], s=3, alpha=0.3, c=C['all_conv'], zorder=1)

    # Overlay removed
    dchi2z_rem = NPZ['dchi2z_removed']
    near_rem = NPZ['near_removed']
    lam_rem = np.abs(NPZ['lam_removed'])
    ax.scatter(dchi2z_rem[near_rem], lam_rem[near_rem], s=8, alpha=0.6,
               c=C['untrim'], zorder=2, label='Removed')
    ax.set_xlabel('Δχ²_z', fontsize=11)
    ax.set_ylabel('|λ|', fontsize=11)
    ax.set_title('B1: |λ| vs Δχ²_z (seed 50)', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)
    rho_s50 = results[0]['b1_rho']
    ax.text(0.98, 0.05, f'ρ = {rho_s50:.3f}', transform=ax.transAxes, fontsize=10,
            ha='right', fontweight='bold', color=C['accent'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4FF', edgecolor=C['accent']))

    # Panel b: B1 ρ across seeds
    ax = axes[1]
    b1_arr = [r['b1_rho'] for r in results if r['b1_rho'] is not None]
    seeds_x = [r['seed_i'] for r in results if r['b1_rho'] is not None]
    ax.scatter(seeds_x, b1_arr, c=C['accent'], s=40, zorder=3, alpha=0.8, edgecolors='white', lw=0.5)
    ax.axhline(0.3, color=C['gold'], ls='--', lw=2, label='B1 threshold (0.3)')
    mean_b1 = np.mean(b1_arr)
    ax.axhline(mean_b1, color=C['trim'], ls='-', lw=2, label=f'Mean = {mean_b1:.4f}')
    ax.set_xlabel('Seed index i', fontsize=11)
    ax.set_ylabel('Spearman ρ(|λ|, Δχ²_z)', fontsize=11)
    ax.set_title('B1: ρ Across 50 Seeds', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)

    # Panel c: B3 AUC across seeds
    ax = axes[2]
    b3_arr = [r['b3_auc'] for r in results if r['b3_auc'] is not None]
    seeds_b3 = [r['seed_i'] for r in results if r['b3_auc'] is not None]
    ax.scatter(seeds_b3, b3_arr, c=C['pink'], s=40, zorder=3, alpha=0.8, edgecolors='white', lw=0.5)
    ax.axhline(0.7, color=C['gold'], ls='--', lw=2, label='B3 threshold (0.7)')
    mean_b3 = np.mean(b3_arr)
    ax.axhline(mean_b3, color=C['derived'], ls='-', lw=2, label=f'Mean = {mean_b3:.4f}')
    ax.set_xlabel('Seed index i', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('B3: AUC Across 50 Seeds', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle('Investigation B — λ–Δχ²_z Correlation and Mechanism',
                 fontsize=14, fontweight='bold', color=C['accent'], y=1.02)
    fig.tight_layout()
    p = os.path.join(_DIR, 'fig2_inv_b_correlation.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return p


def fig3_per_band():
    """Fig 3: Per-band removal rates and C3 nearest-band fraction."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Aggregate per-band rates across seeds
    all_rates = np.array([r['c1_per_band_rates'] for r in results], dtype=float)  # (50, 17)
    mean_rates = np.nanmean(all_rates, axis=0)
    std_rates = np.nanstd(all_rates, axis=0)

    # Panel a: per-band removal rate
    ax = axes[0]
    colors = [C['sensor'] if IS_SENSOR[j] else C['service'] for j in range(17)]
    bars = ax.bar(range(17), mean_rates * 100, yerr=std_rates * 100, color=colors,
                  alpha=0.75, edgecolor='white', lw=1, capsize=3)
    ax.set_xticks(range(17))
    ax.set_xticklabels([f'{int(z)}' for z in BAND_Z], fontsize=6, rotation=45)
    ax.set_xlabel('Band centre z [mm]', fontsize=10)
    ax.set_ylabel('Removal rate [%]', fontsize=10)
    ax.set_title('C1: Per-Band Removal Rate', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(handles=[Patch(facecolor=C['sensor'], alpha=0.75, label='Sensor'),
                        Patch(facecolor=C['service'], alpha=0.75, label='Service')],
              fontsize=9, framealpha=0.9)

    # Panel b: removal rate vs HW (C2)
    ax = axes[1]
    # Average per-band rate across seeds for each band
    for j in range(17):
        c = C['sensor'] if IS_SENSOR[j] else C['service']
        ax.scatter(BAND_HW[j], mean_rates[j] * 100, c=c, s=60, zorder=3,
                   edgecolors='white', lw=0.8)
    ax.set_xlabel('Band half-width HW [mm]', fontsize=11)
    ax.set_ylabel('Mean removal rate [%]', fontsize=11)
    ax.set_title('C2: Rate vs HW', fontsize=12, fontweight='bold', color=C['accent'])
    c2_mean = np.mean([r['c2_rho'] for r in results if r['c2_rho'] is not None])
    ax.text(0.98, 0.05, f'Mean ρ = {c2_mean:.3f}', transform=ax.transAxes, fontsize=10,
            ha='right', fontweight='bold', color=C['accent'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4FF', edgecolor=C['accent']))

    # Panel c: C3 nearest-band fraction
    ax = axes[2]
    c3_arr = [r['c3_mean_frac'] for r in results if r['c3_mean_frac'] is not None]
    ax.hist(c3_arr, bins=20, color=C['trim'], alpha=0.7, edgecolor='white', lw=0.8)
    ax.axvline(0.8, color=C['gold'], ls='--', lw=2, label='C3 threshold (0.8)')
    mean_c3 = np.mean(c3_arr)
    ax.axvline(mean_c3, color=C['accent'], ls='-', lw=2, label=f'Mean = {mean_c3:.4f}')
    ax.set_xlabel('Nearest-band p_conv fraction', fontsize=11)
    ax.set_ylabel('Seeds', fontsize=11)
    ax.set_title('C3: p_conv Dominance', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle('Investigation C — Per-Band Decomposition',
                 fontsize=14, fontweight='bold', color=C['accent'], y=1.02)
    fig.tight_layout()
    p = os.path.join(_DIR, 'fig3_inv_c_bands.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return p


def fig4_decomposition():
    """Fig 4: λ decomposition (D1, D2) and overlap metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: D1 and D2 ratios across seeds
    ax = axes[0]
    d1_arr = [r['d1_ratio'] for r in results if r['d1_ratio'] is not None]
    d2_arr = [r['d2_ratio'] for r in results if r['d2_ratio'] is not None]
    a1_arr = [r['a1_ratio'] for r in results if r['a1_ratio'] is not None]
    seeds_d = [r['seed_i'] for r in results if r['d1_ratio'] is not None]

    ax.scatter(seeds_d, a1_arr[:len(seeds_d)], c=C['accent'], s=35, alpha=0.7, label='Full λ (A1)', zorder=3, edgecolors='white', lw=0.5)
    ax.scatter(seeds_d, d1_arr, c=C['trim'], s=35, alpha=0.7, label='λ_pos (D1)', zorder=3, marker='s', edgecolors='white', lw=0.5)
    ax.scatter(seeds_d, d2_arr, c=C['untrim'], s=35, alpha=0.7, label='λ_fit (D2)', zorder=3, marker='^', edgecolors='white', lw=0.5)
    ax.axhline(2.0, color=C['gold'], ls='--', lw=2, label='Threshold (2.0)')
    ax.set_xlabel('Seed index i', fontsize=11)
    ax.set_ylabel('Mean Δχ²_z ratio', fontsize=11)
    ax.set_title('D1/D2: Decomposed Ratios', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)

    # Panel b: overlap fractions
    ax = axes[1]
    op_u = [r['overlap_pos_upper'] for r in results if r['overlap_pos_upper'] is not None]
    op_l = [r['overlap_pos_lower'] for r in results if r['overlap_pos_lower'] is not None]
    of_u = [r['overlap_fit_upper'] for r in results if r['overlap_fit_upper'] is not None]
    of_l = [r['overlap_fit_lower'] for r in results if r['overlap_fit_lower'] is not None]

    labels = ['λ_pos\nupper', 'λ_pos\nlower', 'λ_fit\nupper', 'λ_fit\nlower']
    means = [np.mean(op_u)*100, np.mean(op_l)*100, np.mean(of_u)*100, np.mean(of_l)*100]
    stds = [np.std(op_u)*100, np.std(op_l)*100, np.std(of_u)*100, np.std(of_l)*100]
    colors_bar = [C['trim'], C['trim'], C['untrim'], C['untrim']]

    bars = ax.bar(range(4), means, yerr=stds, color=colors_bar, alpha=0.75,
                  edgecolor='white', lw=1.5, capsize=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Overlap with full λ [%]', fontsize=11)
    ax.set_title('Overlap with Full λ', fontsize=12, fontweight='bold', color=C['accent'])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold', color=C['text'])

    # Panel c: bar chart of channel strengths
    ax = axes[2]
    mean_a1 = np.mean(a1_arr)
    mean_d1 = np.mean(d1_arr)
    mean_d2 = np.mean(d2_arr)
    bars = ax.bar(['Full λ', 'λ_pos\n(position)', 'λ_fit\n(fit)'],
                  [mean_a1, mean_d1, mean_d2],
                  color=[C['accent'], C['trim'], C['untrim']], alpha=0.8,
                  edgecolor='white', lw=1.5)
    ax.axhline(2.0, color=C['gold'], ls='--', lw=2, label='Threshold (2.0)')
    ax.set_ylabel('Mean Δχ²_z ratio', fontsize=11)
    ax.set_title('Channel Comparison', fontsize=12, fontweight='bold', color=C['accent'])
    ax.legend(fontsize=9, framealpha=0.9)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f'{b.get_height():.2f}', ha='center', fontsize=10, fontweight='bold', color=C['text'])

    fig.suptitle('Investigation D — λ Decomposition Test',
                 fontsize=14, fontweight='bold', color=C['accent'], y=1.02)
    fig.tight_layout()
    p = os.path.join(_DIR, 'fig4_inv_d_decomposition.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return p


def fig5_scorecard():
    """Fig 5: Prediction scorecard."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    n_pred = len(predictions)
    cols = ['ID', 'Description', 'Criterion', 'Tier', 'Measured', 'Result']
    col_x = [0.02, 0.08, 0.38, 0.62, 0.72, 0.88]
    header_y = 0.94
    row_h = 0.075

    # Header
    for cx, label in zip(col_x, cols):
        ax.text(cx, header_y, label, fontsize=10, fontweight='bold', color=C['bg'],
                transform=ax.transAxes, va='center')
    ax.add_patch(plt.Rectangle((0.01, header_y - 0.025), 0.98, 0.05,
                                transform=ax.transAxes, facecolor=C['accent'],
                                clip_on=False, zorder=2))

    n_confirmed = sum(1 for p in predictions if p['result'] == 'CONFIRMED')
    n_core_conf = sum(1 for p in predictions if p['tier'] == 'Core' and p['result'] == 'CONFIRMED')

    for j, pred in enumerate(predictions):
        y = header_y - (j + 1) * row_h
        if pred['result'] == 'CONFIRMED':
            bg_c = '#F0FFF0'
        elif pred['result'] == 'REPORTED':
            bg_c = '#F0F4FF'
        elif pred['result'] == 'NOT EVALUABLE':
            bg_c = '#FFFFF0'
        else:
            bg_c = '#FFF0F0'
        ax.add_patch(plt.Rectangle((0.01, y - 0.025), 0.98, row_h - 0.005,
                                    transform=ax.transAxes, facecolor=bg_c,
                                    edgecolor=C['grid'], lw=0.5, clip_on=False))
        val_str = 'N/A' if pred.get('value') is None else str(pred['value'])
        vals = [pred['id'], pred['desc'][:35], pred['criterion'],
                pred['tier'], val_str, pred['result']]
        for cx, v in zip(col_x, vals):
            if v == 'CONFIRMED':
                clr = C['confirm']
            elif v == 'REFUTED':
                clr = C['refute']
            elif v in ('NOT EVALUABLE', 'REPORTED'):
                clr = C['gold']
            else:
                clr = C['text']
            fw = 'bold' if cx in (col_x[0], col_x[-1]) else 'normal'
            ax.text(cx, y, v, fontsize=8.5, color=clr, fontweight=fw,
                    transform=ax.transAxes, va='center')

    ax.set_title(f'v3.0 Prediction Scorecard  —  {n_confirmed}/{n_pred} CONFIRMED  |  Core: {n_core_conf}/4',
                 fontsize=14, fontweight='bold', color=C['accent'], pad=20)

    strength = summary['overall_strength']
    s_color = C['confirm'] if strength == 'STRONG' else (C['gold'] if strength == 'PARTIAL' else C['refute'])
    ax.text(0.5, 0.02, f'Overall: {strength} support ({n_core_conf}/4 core predictions confirmed)',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color=s_color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8FF', edgecolor=s_color, alpha=0.9))

    fig.tight_layout()
    p = os.path.join(_DIR, 'fig5_prediction_scorecard.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return p


# ════════════════════════════════════════════════════════════
# Generate figures
# ════════════════════════════════════════════════════════════
print("Generating figures...")
FIG1 = fig1_dchi2z_distribution()
FIG2 = fig2_lambda_correlation()
FIG3 = fig3_per_band()
FIG4 = fig4_decomposition()
FIG5 = fig5_scorecard()
print("  Figures done.")

fig_width  = 170*mm
fig_height_std  = 65*mm
fig_height_tall = 80*mm
fig_height_wide = 90*mm

# ════════════════════════════════════════════════════════════
# BUILD PDF
# ════════════════════════════════════════════════════════════

OUT_PDF = os.path.join(_DIR,
    'ALP-ttbar_Toy-MC_v3.0_Post-Registration_DeltaChi2z-Profiling-Results_2026-11-27.pdf')

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                         leftMargin=15*mm, rightMargin=12*mm,
                         topMargin=18*mm, bottomMargin=18*mm)
story = []

# ── Title ──
story.append(Paragraph(
    "ALP-ttbar Toy MC v3.0 — Post-Registration:<br/>"
    "Δχ<super>2</super><sub>z</sub> Profiling of Removed Near-Band Vertices: Results",
    sTitle))
story.append(Paragraph(
    "W. S. Paik  ·  27 November 2026  ·  DOI: 10.5281/zenodo.24581149<br/>"
    "Preregistration: DOI 10.5281/zenodo.24554495",
    sSubTitle))

# ════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════
story.append(Paragraph("1. Executive Summary", sH1))

n_core_conf = summary['n_core_confirmed']
strength = summary['overall_strength']

# Get actual values
a1_val = [p for p in predictions if p['id'] == 'A1'][0]
a2_val = [p for p in predictions if p['id'] == 'A2'][0]
b1_val = [p for p in predictions if p['id'] == 'B1'][0]
d1_val = [p for p in predictions if p['id'] == 'D1'][0]
d2_val = [p for p in predictions if p['id'] == 'D2'][0]

story.append(Paragraph(
    f"This post-registration reports the results of the v3.0 preregistered study "
    f"(DOI 10.5281/zenodo.24554495), which tested whether removed near-band vertices exhibit "
    f"systematically higher {dchiz} (the squared standardised distance from the nearest band "
    f"centre) than surviving vertices. The study ran 50 seeds (i = 50–99) under configuration "
    f"E3 (f = 0.25) and evaluated 11 predictions across four investigations (A–D). "
    f"<b>Overall verdict: {strength} support</b> for the {dchiz} association hypothesis, "
    f"with {n_core_conf}/4 Core predictions confirmed.",
    sBody))

story.append(Paragraph(
    f"<b>Core results.</b> A1 (mean {dchiz} ratio) is strongly confirmed at "
    f"{a1_val['value']:.2f} (threshold: 2.0), indicating that removed vertices have ~4× higher "
    f"mean {dchiz} than surviving near-band vertices. B1 (Spearman ρ between |λ| and {dchiz}) "
    f"is confirmed at {b1_val['value']:.4f} (threshold: 0.3), establishing a monotonic "
    f"relationship. D1 (position-only λ<sub>pos</sub>) is confirmed at {d1_val['value']:.2f} "
    f"(threshold: 2.0), demonstrating that the {psub} channel alone — without the "
    f"Δχ<super>2</super><sub>3D</sub>/2 term — produces elevated {dchiz} in its removed set. "
    f"However, A2 (enrichment at {dchiz} > 4) is <b>refuted</b> at {a2_val['value']:.2f}× "
    f"(threshold: 3.0×), indicating that removed vertices are not concentrated specifically "
    f"in the high-{dchiz} tail despite having elevated mean {dchiz}.",
    sBody))

story.append(Paragraph(
    f"<b>Key insight from decomposition.</b> The D2 fit-only channel (λ<sub>fit</sub> = "
    f"Δχ<super>2</super><sub>3D</sub>/2) shows an even stronger ratio of {d2_val['value']:.2f}, "
    f"and its upper-tail overlap with the full λ removal list is "
    f"{overlap['fit_upper_mean']*100:.1f}% ± {overlap['fit_upper_std']*100:.1f}%, while "
    f"λ<sub>pos</sub> overlap is only {overlap['pos_upper_mean']*100:.1f}% ± "
    f"{overlap['pos_upper_std']*100:.1f}%. This indicates that the Δχ<super>2</super><sub>3D</sub>/2 "
    f"term dominates the upper-tail removal decisions, while both channels contribute to "
    f"lower-tail removal (λ<sub>pos</sub> lower: {overlap['pos_lower_mean']*100:.1f}%, "
    f"λ<sub>fit</sub> lower: {overlap['fit_lower_mean']*100:.1f}%). The {dchiz} association "
    f"measured in A1 and B1 is genuine but largely attributable to the Δχ<super>2</super><sub>3D</sub>/2 "
    f"fit-quality term in λ. This is consistent with the circularity concern "
    f"discussed in §2.3 of the preregistration, though the present analysis does not "
    f"separately decompose Δχ<super>2</super><sub>L</sub> vs Δχ<super>2</super><sub>T</sub> "
    f"within Δχ<super>2</super><sub>3D</sub>.",
    sBody))


# ════════════════════════════════════════════════════════════
# 2. METHODOLOGY
# ════════════════════════════════════════════════════════════
story.append(Paragraph("2. Methodology", sH1))
story.append(Paragraph(
    "All procedures followed the preregistered protocol (DOI 10.5281/zenodo.24554495) without "
    "modification. The frozen simulation modules (endcap_sim.py, mtag_v24.py, mtag_v22.py) "
    "were MD5-verified before execution:",
    sBody))

md5_header = [Paragraph("Module", sTH), Paragraph("MD5 (verified)", sTH), Paragraph("Origin", sTH)]
md5_rows = [
    [Paragraph("endcap_sim.py", sTDL), Paragraph("7daa81b2…8c2d0", sTD), Paragraph("v2.5", sTD)],
    [Paragraph("mtag_v24.py", sTDL), Paragraph("39451a21…54c6e", sTD), Paragraph("v2.4", sTD)],
    [Paragraph("mtag_v22.py", sTDL), Paragraph("a687701e…8c39", sTD), Paragraph("v2.2", sTD)],
]
story.append(make_table(md5_header, md5_rows, [40*mm, 60*mm, 35*mm]))
story.append(Spacer(1, 2*mm))

story.append(Paragraph(
    f"<b>Key definitions.</b> {dchiz} = (Δz/σ<sub>z</sub>)<super>2</super> where "
    f"Δz = |z<sub>m</sub> − Z<sub>nearest</sub>|. Near-band: within d<sub>band</sub> = "
    f"{D_BAND:.1f} mm of any of the 17 band centres. Trimming: k = 1% from each tail of "
    f"signed λ (or λ<sub>pos</sub>, λ<sub>fit</sub>), with ties broken by ascending event "
    f"index. Exactly ⌊0.01 × N<sub>base</sub>⌋ vertices removed per tail.",
    sBody))

story.append(Paragraph(
    f"<b>Configuration.</b> E3: f = 0.25, η = 2.0, 17-band geometry. Seeds i = 50–99 "
    f"(50 seeds). N<sub>sig</sub> ≤ 50,000; N<sub>conv</sub> ≤ 100,000. Focus mass "
    f"m<sub>a</sub> = 17 MeV.",
    sBody))


# ════════════════════════════════════════════════════════════
# 3. RESULTS
# ════════════════════════════════════════════════════════════
story.append(Paragraph("3. Results", sH1))

# --- 3.1 Investigation A ---
story.append(Paragraph("3.1 Investigation A: Δχ<super>2</super><sub>z</sub> Distribution Comparison", sH2))

a1p = [p for p in predictions if p['id'] == 'A1'][0]
a2p = [p for p in predictions if p['id'] == 'A2'][0]
a3p = [p for p in predictions if p['id'] == 'A3'][0]

story.append(Paragraph(
    f"<b>A1 (Core) — CONFIRMED.</b> The mean {dchiz} ratio of removed to surviving near-band "
    f"vertices is {a1p['value']:.2f} ± {a1p['std']:.2f} across 50 seeds (threshold: > 2.0). "
    f"Removed vertices have ~4× higher mean {dchiz}, a substantial and consistent shift. "
    f"All 50 seeds individually exceed the threshold (minimum: "
    f"{min(r['a1_ratio'] for r in results if r['a1_ratio'] is not None):.2f}).",
    sBullet))

story.append(Paragraph(
    f"<b>A2 (Core) — REFUTED.</b> The mean enrichment at {dchiz} > 4 is {a2p['value']:.2f}× "
    f"(threshold: > 3.0×). The surviving near-band fraction above 4 is ~53%, so even if "
    f"100% of removed vertices exceeded the threshold, enrichment would be at most ~1.88×, "
    f"making the 3× target structurally unreachable. The measured fractions (~51.5% removed "
    f"vs ~53.3% surviving) show near-parity, indicating the {dchiz} shift is a broad "
    f"distributional change (elevated mean) rather than tail-concentrated enrichment.",
    sBullet))

story.append(Paragraph(
    f"<b>A3 (Secondary) — CONFIRMED.</b> The coefficient of variation of per-seed mean "
    f"{dchiz} in the removed set is {a3p['value']:.2f}% (threshold: < 50%). The {dchiz} "
    f"shift is a stable structural feature, not driven by outlier seeds.",
    sBullet))

# Figure 1
story.append(Spacer(1, 2*mm))
story.append(Paragraph("<b>Figure 1.</b> Investigation A — Δχ<super>2</super><sub>z</sub> Distribution", sFigLabel))
story.append(Image(FIG1, width=fig_width, height=fig_height_std))
story.append(Paragraph(
    "<b>Figure 1.</b> Left: Δχ²<sub>z</sub> distribution for surviving (grey) and removed "
    "(red) near-band vertices (seed 50). The removed distribution is shifted rightward. "
    "Right: per-seed ratio (mean Δχ²<sub>z</sub> removed / surviving) across 50 seeds; "
    "all seeds exceed the A1 threshold of 2.0 (orange dashed).",
    sCaption))

story.append(PageBreak())

# --- 3.2 Investigation B ---
story.append(Paragraph("3.2 Investigation B: λ–Δχ<super>2</super><sub>z</sub> Correlation", sH2))

b1p = [p for p in predictions if p['id'] == 'B1'][0]
b2p = [p for p in predictions if p['id'] == 'B2'][0]
b3p = [p for p in predictions if p['id'] == 'B3'][0]

story.append(Paragraph(
    f"<b>B1 (Core) — CONFIRMED.</b> The mean Spearman rank correlation between |λ| and "
    f"{dchiz} across near-band vertices is ρ = {b1p['value']:.4f} ± {b1p['std']:.4f} "
    f"(threshold: > 0.3). A moderate monotonic relationship exists: vertices farther from "
    f"band centres (higher {dchiz}) tend to have more extreme |λ| values. As cautioned in "
    f"the preregistration (§2.3), this correlation is partly expected because "
    f"Δχ<super>2</super><sub>L</sub> (a related z-pull metric) feeds into λ via the "
    f"Δχ<super>2</super><sub>3D</sub>/2 term. The decomposition in §3.4 quantifies this.",
    sBullet))

story.append(Paragraph(
    f"<b>B2 (Diagnostic) — REFUTED.</b> The mean asymmetry |Δmean|/pooled = "
    f"{b2p['value']:.4f} (threshold: < 0.2). The upper and lower λ tails show substantially "
    f"different mean {dchiz}: the upper tail (high λ, signal-like) has higher {dchiz} than "
    f"the lower tail (low λ, conversion-like). This asymmetry is physically expected: "
    f"the upper tail removes vertices that look signal-like (far from bands, high {dchiz}), "
    f"while the lower tail removes extreme-conversion vertices (near bands, potentially "
    f"lower {dchiz}).",
    sBullet))

story.append(Paragraph(
    f"<b>B3 (Secondary) — REFUTED.</b> The mean AUC = {b3p['value']:.4f} "
    f"(threshold: > 0.7). {dchiz} alone has limited discriminating power for predicting "
    f"removal, confirming that removal is determined by the full λ — which includes both the "
    f"position log-likelihood and the 3D fit quality — rather than by z-displacement alone.",
    sBullet))

# Figure 2
story.append(Spacer(1, 2*mm))
story.append(Paragraph("<b>Figure 2.</b> Investigation B — λ–Δχ<super>2</super><sub>z</sub> Correlation", sFigLabel))
story.append(Image(FIG2, width=fig_width, height=fig_height_std))
story.append(Paragraph(
    "<b>Figure 2.</b> Left: scatter plot of |λ| vs Δχ²<sub>z</sub> for near-band vertices "
    "(seed 50), with removed vertices highlighted (red). Centre: Spearman ρ across 50 seeds, "
    "all exceeding the B1 threshold of 0.3. Right: AUC from logistic regression using "
    "Δχ²<sub>z</sub> alone; all below the B3 threshold of 0.7.",
    sCaption))


# --- 3.3 Investigation C ---
story.append(Paragraph("3.3 Investigation C: Per-Band Decomposition", sH2))

c1p = [p for p in predictions if p['id'] == 'C1'][0]
c2p = [p for p in predictions if p['id'] == 'C2'][0]
c3p = [p for p in predictions if p['id'] == 'C3'][0]

story.append(Paragraph(
    f"<b>C1 (Diagnostic) — CONFIRMED.</b> The mean CV of per-band removal rates is "
    f"{c1p['value']:.1f}% (threshold: > 20%). Removal is not uniform across bands; "
    f"some bands contribute disproportionately more extreme-λ vertices.",
    sBullet))

story.append(Paragraph(
    f"<b>C2 (Diagnostic) — REFUTED.</b> The mean Spearman ρ between removal rate and "
    f"band half-width is {c2p['value']:.4f} (prediction: ρ < 0). The positive correlation "
    f"indicates that bands with larger spatial half-width HW actually have <i>higher</i> "
    f"removal rates, contrary to the prediction. This may reflect that wider bands tend to "
    f"be outer sensor disks with higher X/X<sub>0</sub> (more material), so the total "
    f"p<sub>conv</sub> contribution — which scales with both HW and c<sub>i</sub> — dominates "
    f"over the geometric spread effect.",
    sBullet))

story.append(Paragraph(
    f"<b>C3 (Secondary) — CONFIRMED.</b> The mean nearest-band p<sub>conv</sub> fraction is "
    f"{c3p['value']:.4f} (threshold: > 0.8). p<sub>conv</sub> is entirely dominated by the "
    f"nearest band's contribution; competing bands play negligible roles. This confirms that "
    f"the z-proximity to a single band drives the {psub} value.",
    sBullet))

# Figure 3
story.append(Spacer(1, 2*mm))
story.append(Paragraph("<b>Figure 3.</b> Investigation C — Per-Band Decomposition", sFigLabel))
story.append(Image(FIG3, width=fig_width, height=fig_height_std))
story.append(Paragraph(
    "<b>Figure 3.</b> Left: per-band removal rates (blue = sensor, orange = service). "
    "Centre: removal rate vs band half-width HW (C2). Right: nearest-band p<sub>conv</sub> "
    "fraction distribution (C3).",
    sCaption))

story.append(PageBreak())

# --- 3.4 Investigation D ---
story.append(Paragraph("3.4 Investigation D: λ Decomposition Test", sH2))

d1p = [p for p in predictions if p['id'] == 'D1'][0]
d2p = [p for p in predictions if p['id'] == 'D2'][0]

story.append(Paragraph(
    f"<b>D1 (Core) — CONFIRMED.</b> The position-only tag λ<sub>pos</sub> (Δχ<super>2</super>"
    f"<sub>3D</sub>/2 set to zero) produces a mean {dchiz} ratio of {d1p['value']:.2f} ± "
    f"{d1p['std']:.2f} (threshold: > 2.0). The {psub} channel alone — without any contribution "
    f"from the 3D fit quality — can produce a removed set with elevated {dchiz}. This "
    f"demonstrates that the association is not purely an artefact of Δχ<super>2</super>"
    f"<sub>3D</sub> embedding.",
    sBullet))

story.append(Paragraph(
    f"<b>D2 (Diagnostic) — REPORTED.</b> The fit-only tag λ<sub>fit</sub> = "
    f"Δχ<super>2</super><sub>3D</sub>/2 produces a mean ratio of {d2p['value']:.2f} ± "
    f"{d2p['std']:.2f}. This is even higher than the full λ ({a1_val['value']:.2f}), which is "
    f"expected because Δχ<super>2</super><sub>3D</sub> contains Δχ<super>2</super><sub>L</sub> "
    f"(the longitudinal z-pull squared), which is directly related to {dchiz}.",
    sBullet))

story.append(Paragraph(
    f"<b>Overlap metrics.</b> The overlap fractions reveal which channel drives the "
    f"original removal:",
    sBody))

ov_header = [Paragraph("Tag", sTH), Paragraph("Upper tail", sTH),
             Paragraph("Lower tail", sTH)]
ov_rows = [
    [Paragraph("λ<sub>pos</sub> (position)", sTDL),
     Paragraph(f"{overlap['pos_upper_mean']*100:.1f}% ± {overlap['pos_upper_std']*100:.1f}%", sTD),
     Paragraph(f"{overlap['pos_lower_mean']*100:.1f}% ± {overlap['pos_lower_std']*100:.1f}%", sTD)],
    [Paragraph("λ<sub>fit</sub> (fit)", sTDL),
     Paragraph(f"{overlap['fit_upper_mean']*100:.1f}% ± {overlap['fit_upper_std']*100:.1f}%", sTD),
     Paragraph(f"{overlap['fit_lower_mean']*100:.1f}% ± {overlap['fit_lower_std']*100:.1f}%", sTD)],
]
story.append(make_table(ov_header, ov_rows, [45*mm, 50*mm, 50*mm]))
story.append(Spacer(1, 2*mm))

story.append(Paragraph(
    f"The upper-tail overlap of λ<sub>fit</sub> with the full λ is "
    f"{overlap['fit_upper_mean']*100:.1f}%, while λ<sub>pos</sub> overlap is only "
    f"{overlap['pos_upper_mean']*100:.1f}%. This demonstrates that the "
    f"Δχ<super>2</super><sub>3D</sub>/2 term almost entirely determines which vertices "
    f"land in the upper 1% tail of the full λ. Conversely, neither channel dominates the "
    f"lower tail (10–15% overlap each), indicating that the lower tail is determined by the "
    f"interaction of both terms. The {psub} channel's independent contribution "
    f"(confirmed by D1 > 2.0) exists but is secondary to the fit-quality channel "
    f"in the full λ's upper-tail removal decisions.",
    sBody))

# Figure 4
story.append(Spacer(1, 2*mm))
story.append(Paragraph("<b>Figure 4.</b> Investigation D — λ Decomposition", sFigLabel))
story.append(Image(FIG4, width=fig_width, height=fig_height_std))
story.append(Paragraph(
    "<b>Figure 4.</b> Left: per-seed Δχ²<sub>z</sub> ratio for the full λ (blue), "
    "λ<sub>pos</sub> (green), and λ<sub>fit</sub> (red). Centre: overlap fractions "
    "with the full λ removal list (per tail). Right: mean ratio comparison across the "
    "three tag variants.",
    sCaption))


# ════════════════════════════════════════════════════════════
# 4. PREDICTION SCORECARD
# ════════════════════════════════════════════════════════════
story.append(Paragraph("4. Prediction Scorecard", sH1))

pred_header = [
    Paragraph("ID", sTH), Paragraph("Description", sTH),
    Paragraph("Criterion", sTH), Paragraph("Tier", sTH),
    Paragraph("Measured", sTH), Paragraph("Result", sTH),
]

pred_rows = []
for p in predictions:
    res_text = p['result']
    res_style = sTDB
    val_str = str(p['value']) if p.get('value') is not None else 'N/A'
    pred_rows.append([
        Paragraph(p['id'], sTDB),
        Paragraph(p['desc'], sTDL),
        Paragraph(p['criterion'], sTD),
        Paragraph(p['tier'], sTDB),
        Paragraph(val_str, sTD),
        Paragraph(res_text, sTDB),
    ])

story.append(Paragraph("<b>Table 1.</b> Prediction Results", sFigLabel))
story.append(make_table(pred_header, pred_rows, [12*mm, 55*mm, 35*mm, 22*mm, 20*mm, 25*mm]))
story.append(Spacer(1, 2*mm))

# Figure 5
story.append(Paragraph("<b>Figure 5.</b> Visual Scorecard", sFigLabel))
story.append(Image(FIG5, width=fig_width, height=fig_height_tall + 10*mm))
story.append(Paragraph(
    f"<b>Figure 5.</b> Visual scorecard of all 11 predictions. Green = CONFIRMED, "
    f"red = REFUTED, blue = REPORTED (diagnostic, no pass/fail). "
    f"Core: {n_core_conf}/4 confirmed → {strength} support.",
    sCaption))


# ════════════════════════════════════════════════════════════
# 5. SUCCESS CRITERIA EVALUATION
# ════════════════════════════════════════════════════════════
story.append(Paragraph("5. Success Criteria Evaluation", sH1))

story.append(Paragraph(
    f"Applying the preregistered tiered criteria: <b>{n_core_conf}/4 Core predictions confirmed "
    f"(A1, B1, D1) → {strength} support</b> for the {dchiz} association hypothesis. "
    f"The refuted A2 prediction indicates that the association is a distributional shift "
    f"(elevated mean) rather than a tail-concentrated enrichment.",
    sBody))

story.append(Paragraph(
    f"• <b>Strong support</b> requires 4/4 Core confirmed. This is not met due to A2 refutation.",
    sBullet))
story.append(Paragraph(
    f"• <b>Partial support</b> requires 2–3/4 Core confirmed. This is met: A1, B1, D1 confirmed.",
    sBullet))

story.append(Paragraph(
    f"<b>Interpretation.</b> The partial-support outcome is scientifically informative. The "
    f"confirmed predictions establish that (i) removed vertices are genuinely shifted toward "
    f"higher {dchiz} (A1); (ii) this shift is monotonically related to |λ| (B1); and (iii) "
    f"the position log-likelihood channel ({psub}) independently contributes to this shift "
    f"(D1), confirming the hypothesis is not purely tautological. The refuted A2 and B3 "
    f"predictions clarify the <i>nature</i> of the association: {dchiz} is elevated in removed "
    f"vertices but does not alone determine removal (AUC = 0.59, well below the full λ's "
    f"discriminating power). The refuted B2 (asymmetry = 1.94) reveals that the two λ tails "
    f"are removed for physically distinct reasons, with the upper tail having much higher "
    f"{dchiz} — consistent with the D-channel decomposition showing that λ<sub>fit</sub> "
    f"dominates upper-tail removal.",
    sBody))


# ════════════════════════════════════════════════════════════
# 6. CONCLUSIONS
# ════════════════════════════════════════════════════════════
story.append(Paragraph("6. Conclusions", sH1))

story.append(Paragraph(
    f"<b>Primary conclusion.</b> The v3.0 study provides partial support for the hypothesis "
    f"that removed near-band vertices have systematically higher {dchiz}. The mean ratio "
    f"of ~4.0 (A1) is well above the threshold, and the effect is stable across all 50 seeds "
    f"(CV = 1.9%, A3). The {psub} position-likelihood channel makes an independent contribution "
    f"(D1 = {d1_val['value']:.2f}), but the fit-quality channel (Δχ<super>2</super><sub>3D</sub>/2) is the "
    f"dominant driver of upper-tail removal, as demonstrated by the 97% upper-tail overlap "
    f"between λ<sub>fit</sub> and the full λ.",
    sBody))

story.append(Paragraph(
    f"<b>The B1 circularity question is partially addressed.</b> The preregistration anticipated that "
    f"the B1 correlation might be partly tautological (§2.3). The decomposition is consistent with this: "
    f"the Δχ<super>2</super><sub>3D</sub>/2 term in λ — which contains z-pull information — "
    f"dominates upper-tail removal. However, the present analysis does not separately decompose "
    f"Δχ<super>2</super><sub>L</sub> vs Δχ<super>2</super><sub>T</sub> within "
    f"Δχ<super>2</super><sub>3D</sub>, so the degree to which Δχ<super>2</super><sub>L</sub> "
    f"specifically drives the correlation remains unquantified. "
    f"D1 confirms that the {psub} channel independently "
    f"produces elevated {dchiz} in its removed set, so the association is not <i>entirely</i> "
    f"circular.",
    sBody))


# ════════════════════════════════════════════════════════════
# 7. FUTURE DIRECTIONS
# ════════════════════════════════════════════════════════════
story.append(Paragraph("7. Future Directions", sH1))
story.append(Paragraph(
    "The v3.0 results — particularly the λ decomposition findings and the refuted predictions "
    "— motivate several concrete follow-up investigations. The first two items below are "
    "identified as priority candidates for a v3.1 preregistration, directly informed by the "
    "results of this study. Subsequent items carry forward the broader research programme "
    "deferred from the v3.0 preregistration (§7.1–§7.6).",
    sBody))

story.append(Paragraph("7.1 λ<sub>pos</sub>-Only Sensitivity Study (v3.1 candidate)", sH2))
story.append(Paragraph(
    f"The decomposition test established that λ<sub>pos</sub> (position-only, with "
    f"Δχ<super>2</super><sub>3D</sub>/2 set to zero) independently produces a removed set "
    f"with elevated {dchiz} (D1 = {d1p['value']:.2f}), yet its upper-tail overlap with the full "
    f"λ is only {overlap['pos_upper_mean']*100:.1f}%. This raises a natural question: how far "
    f"can the pure position-likelihood channel go without the fit-quality term? A dedicated "
    f"study would trim using λ<sub>pos</sub> alone and evaluate the resulting signal efficiency "
    f"and upper-limit proxy, comparing against the full-λ trimming. If λ<sub>pos</sub>-only "
    f"trimming retains comparable sensitivity, it would demonstrate that material-proximity "
    f"information ({psub}) is sufficient for effective background trimming — a finding with "
    f"implications for detectors where vertex-fit quality information may be limited or "
    f"unreliable. Conversely, if sensitivity degrades substantially, it would confirm that the "
    f"Δχ<super>2</super><sub>3D</sub>/2 term's discriminating power is essential and not "
    f"redundant with the position channel.",
    sBody))

story.append(Paragraph("7.2 C2 Material-Budget Dependence (v3.1 candidate)", sH2))
story.append(Paragraph(
    f"The refuted C2 prediction (ρ = +{c2p['value']:.3f}, predicted < 0) revealed that wider "
    f"bands have <i>higher</i> removal rates, contrary to the geometric-spread hypothesis. "
    f"The likely explanation is that wider bands tend to be outer sensor disks with higher "
    f"X/X<sub>0</sub>, so the total {psub} contribution — which scales with both HW and "
    f"c<sub>i</sub> (material coefficient) — dominates over the half-width effect. A systematic "
    f"investigation would disentangle these two drivers by (i) computing the partial correlation "
    f"of removal rate with HW after controlling for c<sub>i</sub>, and (ii) testing whether the "
    f"positive C2 correlation persists under different material scale factors "
    f"f ∈ {{0.10, 0.50, 1.00}}. If the correlation reverses at higher f (where all bands are "
    f"material-rich), this would confirm that the effect is driven by the c<sub>i</sub> hierarchy "
    f"rather than band geometry per se.",
    sBody))

story.append(Paragraph("7.3 Trim-Fraction Optimisation (k &lt; 1%)", sH2))
story.append(Paragraph(
    "The v2.9 result that f<sub>B,rem</sub> = 1.000 at k = 1% suggests that the fixed 1% trim "
    "fraction may be larger than necessary. A systematic scan over k ∈ {0.1%, 0.2%, 0.5%, "
    "0.75%, 1.0%} could identify the minimum trim fraction that maintains 100% B-contributor "
    "capture while improving signal retention. The benefit of a smaller k should be evaluated by "
    "comparing B-capture rate, signal efficiency, and the toy upper-limit proxy max(3, "
    "2√B)/S jointly. This investigation may be informed by the v3.0 finding that "
    f"λ<sub>fit</sub> dominates upper-tail removal: if the Δχ<super>2</super><sub>3D</sub>/2 "
    f"term is the primary driver, a {dchiz}-based threshold might outperform a fixed percentile cut.",
    sBody))

story.append(Paragraph("7.4 Signal-Side λ Regularisation", sH2))
story.append(Paragraph(
    "Although v2.9 found zero floor-applied vertices among the removed conversion-background "
    "population, the same p<sub>conv</sub> floor mechanism may still affect signal vertices. "
    "Regularising p<sub>conv</sub> for signal vertices — imposing a physical minimum above "
    "P<sub>FLOOR</sub> — could reduce extreme signal-vertex λ tail inflation. Since v2.7 found "
    "that signal-vertex randomness contributes only 0.19% of Tag variance, the effect may be "
    "small, but a dedicated test would clarify the scope of any regularisation. Any change to "
    "the tag formula itself would require a fresh preregistration.",
    sBody))

story.append(Paragraph("7.5 Extension to Other Configurations and Mass Grid", sH2))
story.append(Paragraph(
    "The current study uses configuration E3 (f = 0.25) at m<sub>a</sub> = 17 MeV. A systematic "
    "scan over f ∈ {0.10, 0.25, 0.50, 1.00} and a broader mass grid (5–500 MeV) would map "
    "the transition point between 'trimming helps' and 'trimming is inert', as suggested by the "
    "v2.8 finding that trimming is concentrated at low mass. The current generator places "
    "conversion vertices only inside sensor disks; testing inter-disk gap conversions would "
    "require extending the generative model.",
    sBody))

story.append(Paragraph("7.6 Alternative Discriminator Architectures", sH2))
story.append(Paragraph(
    "The v2.9 result that the removal mask captures 100% of B-contributors suggests a targeted "
    "pre-filter approach: rather than trimming a fixed 1% tail, a future discriminator could "
    "directly identify and remove near-band conversion vertices with high "
    f"{dchiz} (as this study confirmed {dchiz} as a correlate of removal). Such a "
    "'material-aware trimming' approach could achieve the benefits of the "
    "current procedure with a physically motivated rather than statistical criterion.",
    sBody))

story.append(Paragraph("7.7 Realistic Validation Programme", sH2))
story.append(Paragraph(
    "The phased validation programme outlined in v2.8 (§10.8) and v2.9 (§9.7–§9.9) — from "
    "event-generator validation (MadGraph5 + Pythia8) through fast detector simulation "
    "(Delphes) to detailed Geant4-based studies — remains a longer-term goal. The v2.9 "
    "finding that all removed vertices are near-band (not floor-artefact) informs this "
    "programme: realistic photon-conversion modelling in Geant4 (where p<sub>conv</sub> is "
    "physics-driven rather than formula-driven) would be the highest-priority validation step.",
    sBody))

story.append(Paragraph("7.8 LHCb VELO as Experimental Test Case", sH2))
story.append(Paragraph(
    "The current LHC Long Shutdown 3 (LS3) and HL-LHC upgrade period may present a timely "
    "opportunity for experimental validation. Among existing LHC experiments, LHCb's forward "
    "disk-based vertex locator (VELO) is a comparable forward detector that could serve as a "
    "test case: its upgraded pixel VELO for Run 3 and beyond provides displaced-vertex "
    "resolution in the forward region. The first step would be to confirm that at 17 MeV the "
    "detector can separate the two tracks from ALP → e<super>+</super>e<super>−</super> and "
    "reconstruct their displaced vertex with sufficient efficiency. If both the track-pair "
    "reconstruction and a common selection rule are validated, the method could be evaluated "
    "in an offline analysis. Forward auxiliary beamline detectors and fixed-target "
    "configurations (SPS/PS energies) could similarly target the 10–50 MeV range, with 17 MeV "
    "as the initial benchmark where the trimming benefit is strongest in this study.",
    sBody))
story.append(Paragraph(
    "Two lines of LHCb precedent are directly relevant. First, the LHCb Collaboration has "
    "constructed a detailed 3-D material map of the VELO using secondary hadronic interactions "
    "from beam-gas collisions [E9]. That study validated the material-interaction classification "
    "procedure using photon conversions to μ<super>+</super>μ<super>−</super> pairs, computing "
    "a p-value for the material-interaction hypothesis at each reconstructed vertex. "
    "This demonstrates that LHCb already possesses "
    "the infrastructure for material-background classification — precisely the task that the "
    "p<sub>conv</sub>-based tag addresses in the toy MC.",
    sBody))
story.append(Paragraph(
    "Second, a recent LHCb analysis [E8] performed the first search for resonances decaying "
    "to photon pairs in a purely neutral final state, using Run 2 data (2018) at "
    "2.1 fb<super>−1</super>. That analysis employed the Run 2 trigger chain — a hardware "
    "trigger followed by two software stages (HLT1, HLT2) — together with LHCb's lower "
    "pile-up and forward geometry, to probe the 4.9–19.4 GeV mass range for ALP → γγ. This "
    "region of parameter space had remained unexplored between the BaBar and ATLAS constraints. "
    "While the γγ channel targets a different ALP coupling and a higher mass scale than the "
    "displaced e<super>+</super>e<super>−</super> signature considered here, the demonstrated "
    "capability of the LHCb trigger and reconstruction infrastructure for unconventional ALP "
    "topologies — requiring a dedicated trigger and selection strategy — suggests that a "
    "displaced-vertex search in the low-mass region could also benefit from the same detector "
    "strengths. The upgraded Run 3 detector, operational since 2022, replaces the hardware "
    "trigger with a fully software-based system, offering further flexibility and larger "
    "datasets.",
    sBody))
story.append(Paragraph(
    "We emphasise that these are extrapolations from a toy-level study; any experimental "
    "application would require the validation programme outlined in §7.7, and the question "
    "of track-pair resolution at the relevant mass scale is one that the detailed simulation "
    "stages are designed to answer. This idea is recorded here to establish a timestamp for "
    "the concept within the preregistered study chain.",
    sBody))


# ════════════════════════════════════════════════════════════
# 8. DOI CHAIN
# ════════════════════════════════════════════════════════════
story.append(Paragraph("8. DOI Chain", sH1))

doi_header = [Paragraph("Version", sTH), Paragraph("DOI", sTH), Paragraph("Description", sTH)]
doi_rows = [
    [Paragraph("<b>v3.0 post-reg</b>", sTDB),
     Paragraph("10.5281/zenodo.24581149", sTD),
     Paragraph("Δχ² Profiling results (this document)", sTDL)],
    [Paragraph("v3.0 prereg", sTD),
     Paragraph("10.5281/zenodo.24554495", sTD),
     Paragraph("Δχ² Profiling prereg", sTDL)],
    [Paragraph("v2.9 post-reg", sTD),
     Paragraph("10.5281/zenodo.24527398", sTD),
     Paragraph("Background-Focused Diagnostics results", sTDL)],
    [Paragraph("v2.9 prereg", sTD),
     Paragraph("10.5281/zenodo.24466187", sTD),
     Paragraph("Background-Focused Diagnostics prereg", sTDL)],
    [Paragraph("v2.8 post-reg", sTD),
     Paragraph("10.5281/zenodo.24437855", sTD),
     Paragraph("Trimmed Tag Estimator results", sTDL)],
    [Paragraph("v2.8 prereg", sTD),
     Paragraph("10.5281/zenodo.24414960", sTD),
     Paragraph("Trimmed Tag Estimator prereg", sTDL)],
    [Paragraph("v2.7 post-reg", sTD),
     Paragraph("10.5281/zenodo.24408990", sTD),
     Paragraph("Tag Formula Diagnostics results", sTDL)],
    [Paragraph("v2.7 prereg", sTD),
     Paragraph("10.5281/zenodo.24388228", sTD),
     Paragraph("Tag Formula Diagnostics prereg", sTDL)],
    [Paragraph("v2.6 post-reg", sTD),
     Paragraph("10.5281/zenodo.24373115", sTD),
     Paragraph("Multi-Seed Ensemble results", sTDL)],
    [Paragraph("v2.6 prereg", sTD),
     Paragraph("10.5281/zenodo.24361872", sTD),
     Paragraph("Multi-Seed Ensemble prereg", sTDL)],
    [Paragraph("v2.5 results", sTD),
     Paragraph("10.5281/zenodo.24341050", sTD),
     Paragraph("Deconflation post-reg", sTDL)],
    [Paragraph("v2.5 prereg", sTD),
     Paragraph("10.5281/zenodo.24329044", sTD),
     Paragraph("Deconflation prereg", sTDL)],
    [Paragraph("v2.4 results", sTD),
     Paragraph("10.5281/zenodo.24302619", sTD),
     Paragraph("17-band geometry post-reg", sTDL)],
    [Paragraph("v2.4 prereg", sTD),
     Paragraph("10.5281/zenodo.24296796", sTD),
     Paragraph("17-band geometry prereg", sTDL)],
    [Paragraph("v2.3 results", sTD),
     Paragraph("10.5281/zenodo.24273586", sTD),
     Paragraph("Endcap tag post-reg", sTDL)],
    [Paragraph("v2.3 prereg", sTD),
     Paragraph("10.5281/zenodo.24244893", sTD),
     Paragraph("Endcap tag prereg", sTDL)],
    [Paragraph("v2.2 results", sTD),
     Paragraph("10.5281/zenodo.24238224", sTD),
     Paragraph("Barrel → endcap post-reg", sTDL)],
    [Paragraph("v2.2 prereg", sTD),
     Paragraph("10.5281/zenodo.24189990", sTD),
     Paragraph("Barrel → endcap prereg", sTDL)],
    [Paragraph("v2.1 results", sTD),
     Paragraph("10.5281/zenodo.24136462", sTD),
     Paragraph("Barrel material-veto post-reg", sTDL)],
    [Paragraph("v2.1 prereg", sTD),
     Paragraph("10.5281/zenodo.24063091", sTD),
     Paragraph("Barrel material-veto prereg", sTDL)],
    [Paragraph("v2.0", sTD),
     Paragraph("10.5281/zenodo.24056097", sTD),
     Paragraph("Initial barrel study", sTDL)],
]
story.append(make_table(doi_header, doi_rows, [30*mm, 55*mm, 80*mm]))


# ════════════════════════════════════════════════════════════
# 8. REFERENCES
# ════════════════════════════════════════════════════════════
story.append(Paragraph("9. References", sH1))
story.append(Paragraph("9.1 External References", sH2))

refs_ext = [
    "[E1] J. Alimena et al., arXiv:2610.38932 [hep-ph] (2026).",
    "[E2] L. Rygaard et al., arXiv:2306.08686 [hep-ph] (2023).",
    "[E3] A. L. Read, J. Phys. G 28 (2002) 2693.",
    "[E4] D. Curtin et al., Rep. Prog. Phys. 82 (2019) 116201.",
    "[E5] NumPy v1.26, numpy.org.",
    "[E6] ReportLab v5.0.1, reportlab.com.",
    "[E7] SciPy v1.14, scipy.org; scikit-learn v1.5, scikit-learn.org.",
    "[E8] LHCb Collaboration, arXiv:2507.14390 [hep-ex] (2025).",
    "[E9] M. Alexander et al., JINST 13 (2018) P06008.",
]
for ref in refs_ext:
    story.append(Paragraph(ref, sBody))

story.append(Paragraph("9.2 Internal References (ALP-ttbar Toy MC Series)", sH2))

refs_int = [
    "[I1] W. S. Paik, v3.0 prereg, DOI: 10.5281/zenodo.24554495 (this study's preregistration).",
    "[I2] W. S. Paik, v2.9 post-reg, DOI: 10.5281/zenodo.24527398 (refuted floor-artefact "
    "hypothesis; identified Δχ² candidate).",
    "[I3] W. S. Paik, v2.9 prereg, DOI: 10.5281/zenodo.24466187.",
    "[I4] W. S. Paik, v2.8 post-reg, DOI: 10.5281/zenodo.24437855 (confirmed trimmed Tag).",
    "[I5] W. S. Paik, v2.8 prereg, DOI: 10.5281/zenodo.24414960.",
    "[I6] W. S. Paik, v2.7 post-reg, DOI: 10.5281/zenodo.24408990.",
]
for ref in refs_int:
    story.append(Paragraph(ref, sBody))

# Footer
story.append(Spacer(1, 6*mm))
story.append(HRFlowable(width="100%", thickness=0.5, color=GREY))
story.append(Paragraph(
    "Generated from frozen simulation modules archived in DOI 10.5281/zenodo.24437855. "
    "Analysis followed preregistered plan DOI 10.5281/zenodo.24554495 without modification. "
    "Post-registration: DOI 10.5281/zenodo.24581149.",
    sFooter))


# ════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════
print(f"Building PDF: {OUT_PDF}")
doc.build(story)
print(f"Done. Output: {OUT_PDF}")
