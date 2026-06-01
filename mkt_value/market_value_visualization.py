"""
Market Value Visualization Script
==================================
Reads players.csv, classifies players as Defensive or Offensive (excluding Goalkeepers),
groups by percentile intervals (10% and 20%), calculates mean market value per group,
and generates bar charts with 95% confidence interval error bars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('mkt_value/players.csv')

# Filter out Goalkeepers and Missing positions, and rows with NaN market values
df = df[~df['position'].isin(['Goalkeeper', 'Missing'])].copy()
df = df.dropna(subset=['market_value_in_eur']).copy()
print(f"Total non-GK players (with valid market values): {len(df)}")

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

print(f"Defensive players: {(df['category']=='Defensive').sum()}")
print(f"Offensive players: {(df['category']=='Offensive').sum()}")

# Check market_value range
print(f"\nMarket value range: €{df['market_value_in_eur'].min():,} – €{df['market_value_in_eur'].max():,}")
print(f"Median: €{df['market_value_in_eur'].median():,}")

# ── 3. Percentile Grouping Function ──────────────────────────────────────────
def percentile_groups(series, n_groups=10):
    """
    Divide a sorted series into n_groups equal-sized percentile bins.
    Returns bin labels (0-indexed) and the actual percentile boundaries.
    """
    sorted_vals = series.sort_values().values
    n = len(sorted_vals)
    bin_size = n / n_groups
    
    # Assign bin indices
    bins = np.floor(np.arange(n) / bin_size).astype(int)
    bins = np.clip(bins, 0, n_groups - 1)
    
    return bins, sorted_vals

def compute_group_stats(category_df, n_groups=10):
    """
    For a single category DataFrame, sort by market_value_in_eur,
    split into n_groups percentile bins, compute mean and Q1–Q3 (IQR) range per bin.
    """
    series = category_df['market_value_in_eur']
    sorted_df = category_df.loc[series.sort_values().index].reset_index(drop=True)
    
    bins, sorted_vals = percentile_groups(series, n_groups)
    sorted_df['percentile_group'] = bins
    
    pct_step = 100 // n_groups
    results = []
    for g in range(n_groups):
        group = sorted_df[sorted_df['percentile_group'] == g]
        vals = group['market_value_in_eur']
        mean_val = vals.mean()
        n_val = len(vals)
        
        # Q1 (25th) and Q3 (75th) percentiles = interquartile range (IQR)
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        
        # Error bar distances from mean to Q1 (lower) and mean to Q3 (upper)
        # Clamp to non-negative (for highly skewed groups where mean may exceed Q3)
        lower_err = max(0, mean_val - q1)
        upper_err = max(0, q3 - mean_val)
        
        # Percentile range for labeling
        p_low = g * pct_step
        p_high = (g + 1) * pct_step
        
        results.append({
            'group': g + 1,
            'percentile_range': f'{p_low}-{p_high}%',
            'mean': mean_val,
            'lower_err': lower_err,
            'upper_err': upper_err,
            'n': n_val,
            'q1': q1,
            'q3': q3,
        })
    
    return pd.DataFrame(results)

# ── 4. Compute Statistics ────────────────────────────────────────────────────
defensive = df[df['category'] == 'Defensive']
offensive = df[df['category'] == 'Offensive']

# 10% groups (10 per category = 20 bars)
def_stats_10 = compute_group_stats(defensive, n_groups=10)
off_stats_10 = compute_group_stats(offensive, n_groups=10)

# 20% groups (5 per category = 10 bars)
def_stats_20 = compute_group_stats(defensive, n_groups=5)
off_stats_20 = compute_group_stats(offensive, n_groups=5)

# Print summary tables
print("\n═══ 10% Groups (Defensive) ═══")
print(def_stats_10[['group', 'percentile_range', 'mean', 'q1', 'q3', 'n']].to_string(index=False))
print("\n═══ 10% Groups (Offensive) ═══")
print(off_stats_10[['group', 'percentile_range', 'mean', 'q1', 'q3', 'n']].to_string(index=False))

print("\n═══ 20% Groups (Defensive) ═══")
print(def_stats_20[['group', 'percentile_range', 'mean', 'q1', 'q3', 'n']].to_string(index=False))
print("\n═══ 20% Groups (Offensive) ═══")
print(off_stats_20[['group', 'percentile_range', 'mean', 'q1', 'q3', 'n']].to_string(index=False))

# ── 5. Plotting Function ─────────────────────────────────────────────────────
def plot_percentile_barchart(def_stats, off_stats, n_groups, filename, add_zoom_inset=False, figsize=(14, 7), title=None):
    """
    Create a grouped bar chart showing mean market value per percentile group,
    with error bars showing Q1–Q3 (IQR) range. Defensive and Offensive bars
    are side-by-side.
    
    Parameters
    ----------
    title : str or None
        Custom title (two lines separated by \\n). If None, auto-generates
        from n_groups.
    add_zoom_inset : bool
        If True, adds an inset axes zooming into the lower-value groups (1–8).
    """
    bar_width = 0.35
    n_groups_percent = 100 // n_groups
    
    # Group labels: use percentile_range from stats if available, else auto-generate
    if 'percentile_range' in def_stats.columns:
        group_labels = def_stats['percentile_range'].tolist()
    else:
        group_labels = [f'{i*n_groups_percent}-{(i+1)*n_groups_percent}%'
                        for i in range(n_groups)]
    
    # Positions
    x = np.arange(n_groups)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    def_color = '#0000a2'   # blue for defensive
    off_color = '#bc272d'   # red for offensive
    
    # Asymmetric error bars: [lower_err, upper_err]
    def_yerr = np.array([def_stats['lower_err'], def_stats['upper_err']])
    off_yerr = np.array([off_stats['lower_err'], off_stats['upper_err']])
    
    # Defensive bars
    bars1 = ax.bar(x - bar_width/2, def_stats['mean'], bar_width,
                   label='Defensive', color=def_color, alpha=0.85,
                   yerr=def_yerr, capsize=3,
                   error_kw={'linewidth': 1.2, 'ecolor': 'dimgrey'})
    
    # Offensive bars
    bars2 = ax.bar(x + bar_width/2, off_stats['mean'], bar_width,
                   label='Offensive', color=off_color, alpha=0.85,
                   yerr=off_yerr, capsize=3,
                   error_kw={'linewidth': 1.2, 'ecolor': 'dimgrey'})
    
    # ── Sample size annotations above bars ────────────────────────────────────
    for i, (d_val, o_val, d_n, o_n) in enumerate(zip(
            def_stats['mean'], off_stats['mean'],
            def_stats['n'], off_stats['n'])):
        # Defensive bar n
        ax.text(x[i] - bar_width/2, d_val + 0.02 * d_val,
                f'n={int(d_n)}', ha='center', va='bottom',
                fontsize=6, color=def_color, rotation=90)
        # Offensive bar n
        ax.text(x[i] + bar_width/2, o_val + 0.02 * o_val,
                f'n={int(o_n)}', ha='center', va='bottom',
                fontsize=6, color=off_color, rotation=90)
    
    # ── Zoom Inset (for 10% chart, groups 1–8) ──────────────────────────────
    if add_zoom_inset and n_groups == 10:
        # Inset axes: positioned at ~20% from left, vertically centered
        # [left, bottom, width, height] in axes coordinates
        ax_inset = fig.add_axes([0.20, 0.31, 0.35, 0.38])
        
        # Groups to show in inset: 1–8 (0-indexed: 0–7)
        zoom_groups = 8
        x_zoom = np.arange(zoom_groups)
        
        # Y-axis limit for zoom: fixed at €1.2M for good visibility
        y_max_zoom = 1_200_000
        
        # Defensive bars in inset
        def_yerr_zoom = np.array([def_stats['lower_err'].iloc[:zoom_groups],
                                   def_stats['upper_err'].iloc[:zoom_groups]])
        ax_inset.bar(x_zoom - bar_width/2, def_stats['mean'].iloc[:zoom_groups],
                     bar_width, color=def_color, alpha=0.85,
                     yerr=def_yerr_zoom, capsize=2,
                     error_kw={'linewidth': 1.0, 'ecolor': 'dimgrey'})
        
        # Offensive bars in inset
        off_yerr_zoom = np.array([off_stats['lower_err'].iloc[:zoom_groups],
                                   off_stats['upper_err'].iloc[:zoom_groups]])
        ax_inset.bar(x_zoom + bar_width/2, off_stats['mean'].iloc[:zoom_groups],
                     bar_width, color=off_color, alpha=0.85,
                     yerr=off_yerr_zoom, capsize=2,
                     error_kw={'linewidth': 1.0, 'ecolor': 'dimgrey'})
        
        # Sample size annotations in inset
        for j in range(zoom_groups):
            d_val = def_stats['mean'].iloc[j]
            o_val = off_stats['mean'].iloc[j]
            d_n = int(def_stats['n'].iloc[j])
            o_n = int(off_stats['n'].iloc[j])
            ax_inset.text(x_zoom[j] - bar_width/2, d_val + 0.02 * d_val,
                          f'n={d_n}', ha='center', va='bottom',
                          fontsize=5, color=def_color, rotation=90)
            ax_inset.text(x_zoom[j] + bar_width/2, o_val + 0.02 * o_val,
                          f'n={o_n}', ha='center', va='bottom',
                          fontsize=5, color=off_color, rotation=90)
        
        ax_inset.set_xticks(x_zoom)
        ax_inset.set_xticklabels(group_labels[:zoom_groups], fontsize=7, rotation=30)
        ax_inset.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f'€{v:,.0f}'))
        ax_inset.tick_params(axis='y', labelsize=7)
        ax_inset.set_ylim(0, y_max_zoom)
        ax_inset.set_title('Zoom: Groups 1–8', fontsize=10, fontweight='bold')
        ax_inset.grid(axis='y', alpha=0.2, linestyle='--')
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
    
    # ── Formatting ──────────────────────────────────────────────────────────
    ax.set_xlabel('Market Value Percentile Group', fontsize=13, labelpad=10)
    ax.set_ylabel('Mean Market Value (EUR)', fontsize=13, labelpad=10)
    if title is None:
        title_pct = n_groups_percent
        title_text = f'Mean Market Value by {title_pct}% Percentile Groups\nDefensive vs Offensive Players'
    else:
        title_text = title
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f'€{v:,.0f}'))
    
    # Create legend with note about error bar meaning
    from matplotlib.lines import Line2D
    total_n = int(def_stats['n'].sum() + off_stats['n'].sum())
    def_total_n = int(def_stats['n'].sum())
    off_total_n = int(off_stats['n'].sum())
    legend_elements = [
        Line2D([0], [0], color=def_color, lw=0, marker='s', markersize=12,
               label=f'Defensive (n={def_total_n})', markerfacecolor=def_color),
        Line2D([0], [0], color=off_color, lw=0, marker='s', markersize=12,
               label=f'Offensive (n={off_total_n})', markerfacecolor=off_color),
        Line2D([0], [0], color='dimgrey', lw=0, marker='_', markersize=12,
               label='Q1–Q3 (IQR)'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Data source attribution
    ax.text(0.5, -0.12, 'Football Data from Transfermarkt by David Cariboo (Kaggle), Public Domain',
            transform=ax.transAxes, fontsize=8, ha='center', va='top',
            color='grey', style='italic')
    
    if add_zoom_inset:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ── 6. Generate Charts ───────────────────────────────────────────────────────
plot_percentile_barchart(def_stats_10, off_stats_10, n_groups=10,
                         filename='mkt_value/market_value_by_10pct_groups.png',
                         add_zoom_inset=True)

plot_percentile_barchart(def_stats_20, off_stats_20, n_groups=5,
                         filename='mkt_value/market_value_by_20pct_groups.png')

# ── 7. Upper-half Chart (50%–100%, no zoom) ─────────────────────────────────
# Slice from index 5 onwards (50-60%, 60-70%, 70-80%, 80-90%, 90-100%)
def_stats_upper = def_stats_10.iloc[5:].reset_index(drop=True)
off_stats_upper = off_stats_10.iloc[5:].reset_index(drop=True)
# Re-number groups for display
def_stats_upper['group'] = range(1, 6)
off_stats_upper['group'] = range(1, 6)

plot_percentile_barchart(def_stats_upper, off_stats_upper, n_groups=5,
                         filename='mkt_value/market_value_upper_50_100pct.png',
                         figsize=(7, 8),
                         title='Mean Market Value (Upper 50–100% Groups)\nDefensive vs Offensive Players')

# ── 7. Also save data tables for reference ────────────────────────────────────
def_stats_10.to_csv('mkt_value/defensive_10pct_groups.csv', index=False)
off_stats_10.to_csv('mkt_value/offensive_10pct_groups.csv', index=False)
def_stats_20.to_csv('mkt_value/defensive_20pct_groups.csv', index=False)
off_stats_20.to_csv('mkt_value/offensive_20pct_groups.csv', index=False)

print("\n✅ All charts and data tables saved successfully!")
print("Files generated:")
print("  - mkt_value/market_value_by_10pct_groups.png  (20 bars)")
print("  - mkt_value/market_value_by_20pct_groups.png  (10 bars)")
print("  - mkt_value/defensive_10pct_groups.csv")
print("  - mkt_value/offensive_10pct_groups.csv")
print("  - mkt_value/defensive_20pct_groups.csv")
print("  - mkt_value/offensive_20pct_groups.csv")
