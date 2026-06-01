"""
Age vs Market Value Curve
=========================
Scatter plot of age vs market value for Defensive and Offensive players,
with LOWESS smoothed trend curves showing the peak value age for each group.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('mkt_value/players.csv')

# Filter out Goalkeepers and Missing positions, NaN market values
df = df[~df['position'].isin(['Goalkeeper', 'Missing'])].copy()
df = df.dropna(subset=['market_value_in_eur']).copy()

# ── 2. Classify: Defensive vs Offensive ───────────────────────────────────────
def classify_position(row):
    pos = row['position']
    sub = row['sub_position']
    if pos == 'Defender':
        return 'Defensive'
    if pos == 'Attack':
        return 'Offensive'
    if pos == 'Midfield':
        if sub == 'Defensive Midfield':
            return 'Defensive'
        else:
            return 'Offensive'
    return None

df['category'] = df.apply(classify_position, axis=1)
df = df.dropna(subset=['category']).reset_index(drop=True)

# ── 3. Calculate Age ─────────────────────────────────────────────────────────
# Extract birth year from date_of_birth (format: "YYYY-MM-DD HH:MM:SS")
df['birth_year'] = pd.to_datetime(df['date_of_birth'], errors='coerce').dt.year
df['age'] = df['last_season'] - df['birth_year']

# Filter out unreasonable ages
df = df[(df['age'] >= 15) & (df['age'] <= 45)].copy()
print(f"Players after age filter: {len(df)}")

# ── 4. Prepare Data ──────────────────────────────────────────────────────────
defensive = df[df['category'] == 'Defensive'].copy()
offensive = df[df['category'] == 'Offensive'].copy()

print(f"Defensive: {len(defensive)}, Offensive: {len(offensive)}")
print(f"Age range: {df['age'].min():.0f} – {df['age'].max():.0f}")

# ── 5. LOWESS Smoothing ──────────────────────────────────────────────────────
def smooth_curve(data, frac=0.4):
    """Apply LOWESS smoothing to age vs market_value data."""
    ages = data['age'].values
    values = data['market_value_in_eur'].values
    # LOWESS returns sorted by x
    smoothed = lowess(values, ages, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]

# Smooth curves
def_ages_sorted = np.sort(defensive['age'].values)
off_ages_sorted = np.sort(offensive['age'].values)

# Use LOWESS
frac = 0.35  # smoothing parameter (lower = more wiggly, higher = smoother)
def_smooth_x, def_smooth_y = smooth_curve(defensive, frac=frac)
off_smooth_x, off_smooth_y = smooth_curve(offensive, frac=frac)

# Find peak age for each group
def_peak_idx = np.argmax(def_smooth_y)
off_peak_idx = np.argmax(off_smooth_y)
def_peak_age = def_smooth_x[def_peak_idx]
def_peak_val = def_smooth_y[def_peak_idx]
off_peak_age = off_smooth_x[off_peak_idx]
off_peak_val = off_smooth_y[off_peak_idx]

print(f"\nDefensive peak: age {def_peak_age:.1f}, value €{def_peak_val:,.0f}")
print(f"Offensive peak: age {off_peak_age:.1f}, value €{off_peak_val:,.0f}")

# ── 6. Age Binned Stats (for the shaded IQR band) ────────────────────────────
def compute_age_bin_stats(data, bin_width=1):
    """Group by age bins and compute mean, Q1, Q3."""
    data = data.copy()
    data['age_bin'] = np.floor(data['age'] / bin_width) * bin_width
    stats = data.groupby('age_bin')['market_value_in_eur'].agg(
        mean='mean', q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n='count'
    ).reset_index()
    return stats

def_bin_stats = compute_age_bin_stats(defensive, bin_width=1)
off_bin_stats = compute_age_bin_stats(offensive, bin_width=1)

# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

colors = {'Defensive': '#0000a2', 'Offensive': '#bc272d'}

# Scatter plot (raw data) — use hexbin for dense regions
# For scatter, downsample to 5000 points per category for performance
def_sample = defensive.sample(n=min(5000, len(defensive)), random_state=42)
off_sample = offensive.sample(n=min(5000, len(offensive)), random_state=42)

ax.scatter(def_sample['age'], def_sample['market_value_in_eur'],
           c=colors['Defensive'], alpha=0.15, s=8, label='_nolegend_')
ax.scatter(off_sample['age'], off_sample['market_value_in_eur'],
           c=colors['Offensive'], alpha=0.15, s=8, label='_nolegend_')

# Shaded IQR bands (per age bin)
ax.fill_between(def_bin_stats['age_bin'], def_bin_stats['q1'], def_bin_stats['q3'],
                color=colors['Defensive'], alpha=0.12, label='_nolegend_')
ax.fill_between(off_bin_stats['age_bin'], off_bin_stats['q1'], off_bin_stats['q3'],
                color=colors['Offensive'], alpha=0.12, label='_nolegend_')

# LOWESS smoothed curves
ax.plot(def_smooth_x, def_smooth_y, color=colors['Defensive'], linewidth=2.5,
        label='Defensive', zorder=5)
ax.plot(off_smooth_x, off_smooth_y, color=colors['Offensive'], linewidth=2.5,
        label='Offensive', zorder=5)

# Mark peak points
ax.scatter([def_peak_age], [def_peak_val], color=colors['Defensive'], s=120,
           marker='*', edgecolors='white', linewidth=1.5, zorder=10,
           label=f"Defensive peak: age {def_peak_age:.0f} (€{def_peak_val/1e6:.1f}M)")
ax.scatter([off_peak_age], [off_peak_val], color=colors['Offensive'], s=120,
           marker='*', edgecolors='white', linewidth=1.5, zorder=10,
           label=f"Offensive peak: age {off_peak_age:.0f} (€{off_peak_val/1e6:.1f}M)")

# Formatting
ax.set_xlabel('Age', fontsize=13, labelpad=10)
ax.set_ylabel('Market Value (EUR)', fontsize=13, labelpad=10)
ax.set_title('Age vs Market Value — Defensive vs Offensive Players\n(LOWESS smoothed curve with Q1–Q3 shaded band)',
             fontsize=14, fontweight='bold', pad=15)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f'€{v:,.0f}'))
ax.set_xlim(15, 42)

ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Data source attribution
ax.text(0.5, -0.10, 'Football Data from Transfermarkt by David Cariboo (Kaggle), Public Domain',
        transform=ax.transAxes, fontsize=8, ha='center', va='top',
        color='grey', style='italic')

plt.tight_layout()
plt.savefig('mkt_value/age_vs_market_value.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: mkt_value/age_vs_market_value.png")
