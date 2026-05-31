#!/usr/bin/env python3
"""Generate the data cleaning & visualization notebook."""
import nbformat as nbf

BASE = '/Users/huangwenqin/Desktop/AIF+CALC+SSF PROJs/Social_Science_Fair'

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

cells = []

# ──────── Helper to avoid f-string vs regular-string confusion ────────
def code(source: str):
    """Add a code cell.  Source is written verbatim (no f-string expansion)."""
    cells.append(nbf.v4.new_code_cell(source))

def markdown(source: str):
    cells.append(nbf.v4.new_markdown_cell(source))

# ══════════════════════════════════════════════════════════════════════
#  CELL 1 — Title
# ══════════════════════════════════════════════════════════════════════
markdown(f"""# Social Science Fair — Sentiment Data Cleaning & Visualisation

**Project root:** `{BASE}/`

This notebook:
1. Loads **all** analysed JSONL files from `Computation/analyzed/`
2. Cleans / filters the data (remove false positives, keep only player mentions)
3. Maps raw position labels → Attack / Defense / Goalkeeper / Midfielder
4. Computes **mention counts**, **mean sentiment** (with 95% CI) per category per team/league
5. Performs **ANOVA + pairwise t-tests** (Bonferroni corrected) per team
6. Produces **grouped bar charts** (sentiment mean with CI; mention count)
7. Produces a **time‑series** plot for club subreddits only (no leagues)
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 2 — Imports & config
# ══════════════════════════════════════════════════════════════════════
code(f"""import json, os, sys
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

BASE = '{BASE}'
DATA_DIR = Path(BASE) / 'Computation' / 'analyzed'
OUT_DIR  = Path(BASE) / 'Computation'
os.chdir(BASE)

print(f'BASE      : {{BASE}}')
print(f'DATA_DIR  : {{DATA_DIR}}')
print(f'OUT_DIR   : {{OUT_DIR}}')

plt.rcParams.update({{
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 150,
}})
print('All imports OK')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 3 — Position → Category mapping
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  Position-to-Category Mapping
#  Attack / Defense / Goalkeeper / Midfielder
# ═══════════════════════════════════════════════════════════

ATTACK_KW = {
    'forward', 'attacker', 'striker', 'center forward',
    'winger', 'left winger', 'right winger', 'wide forward',
    'left wing', 'right wing',
    'attacking midfielder', 'attacking-midfielder',
    'central attacking midfielder',
    'forward forward',
}

DEFENSE_KW = {
    'defender', 'wide defender',
    'fullback', 'full back', 'full-back',
    'left back', 'leftback', 'left-back',
    'right back', 'rightback', 'right-back',
    'centre back', 'center back', 'centreback', 'centerback',
    'centre-back', 'center-back',
    'defensive midfielder', 'defensive-midfielder',
    'defensive midfield',
    'center defensive midfielder',
    'wingback', 'wing back', 'wing-back',
    'central midfielder',
}

GK_KW = {'goalkeeper', 'keeper'}

MIDFIELDER_KW = {
    'midfielder',
    'left midfielder', 'right midfielder',
    'centre midfielder', 'center midfielder',
    'centre-midfielder',
}

def pos_to_cat(pos) -> str | None:
    if not pos or not isinstance(pos, str):
        return None
    pos = pos.strip()
    if not pos:
        return None
    # comma-separated multi-position → take first
    if ',' in pos:
        pos = pos.split(',')[0].strip()
    # normalise
    p = pos.lower().replace('-', ' ').replace('_', ' ')
    p = ' '.join(p.split())

    # exact match
    if p in ATTACK_KW:
        return 'Attack'
    if p in DEFENSE_KW:
        return 'Defense'
    if p in GK_KW:
        return 'Goalkeeper'
    if p in MIDFIELDER_KW:
        return 'Midfielder'

    # substring fallback
    words = set(p.split())
    if words & {'forward', 'striker', 'winger', 'attacker'}:
        return 'Attack'
    if 'defender' in words or 'fullback' in words:
        return 'Defense'
    if 'keeper' in words:
        return 'Goalkeeper'
    if 'midfielder' in words:
        return 'Midfielder'
    return None  # Manager, Coach, Referee, N/A, Unknown, …

# Quick validation
tests = [
    ('Forward', 'Attack'), ('Defender', 'Defense'), ('Goalkeeper', 'Goalkeeper'),
    ('Manager', None), ('N/A', None), ('Winger', 'Attack'),
    ('Attacking Midfielder', 'Attack'), ('Fullback', 'Defense'),
    ('Defensive Midfielder', 'Defense'), ('Striker', 'Attack'),
    ('Right Back', 'Defense'), ('Left Back', 'Defense'),
    ('Centre Back', 'Defense'), ('Unknown', None),
    ('Retired Defender', 'Defense'), ('Former Forward', 'Attack'),
    ('Forward, Forward', 'Attack'), ('Forward, Midfielder, Midfielder', 'Attack'),
    ('Manager, Defender', None),
    ('Center Back', 'Defense'), ('Right-back', 'Defense'),
    ('Wingback', 'Defense'), ('Full-Back', 'Defense'),
    ('Center Forward', 'Attack'), ('Central Midfielder', 'Defense'),
    ('Midfielder', 'Midfielder'),
    ('Left Midfielder', 'Midfielder'), ('Right Midfielder', 'Midfielder'),
    ('Centre-midfielder', 'Midfielder'), ('Centre-Midfielder', 'Midfielder'),
    ('', None), ('None', None), ('none', None),
]
all_ok = True
for inp, exp in tests:
    got = pos_to_cat(inp)
    ok = '✓' if got == exp else '✗'
    if got != exp:
        all_ok = False
    print(f'  {ok}  {inp:35s} -> {str(got):12s}  (expected {str(exp)})')
print(f'\nAll tests passed: {all_ok}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 4 — Load & flatten
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  Load & flatten every JSONL file
# ═══════════════════════════════════════════════════════════

TEAM_LABELS = {
    'Barca': 'Barcelona', 'Bundesliga': 'Bundesliga',
    'fcbayern': 'FC Bayern', 'LaLiga': 'LaLiga', 'Ligue1': 'Ligue1',
    'LiverpoolFC': 'Liverpool', 'ManchesterUnited': 'Man United',
    'PremierLeague': 'Premier League', 'realmadrid': 'Real Madrid',
}
LEAGUE_SUBS = {'Bundesliga', 'LaLiga', 'Ligue1', 'PremierLeague'}

jsonl_files = sorted(DATA_DIR.glob('*.jsonl'))
print(f'Found {len(jsonl_files)} files')

total_lines = sum(sum(1 for _ in open(f, encoding='utf-8')) for f in jsonl_files)
print(f'Total lines: {total_lines:,}')

records = []
post_count_per_team = {}  # subreddit -> total post count (after FP+no_groups filter)
skip_fp = skip_ng = skip_cat = skip_sent = 0

pbar = tqdm(total=total_lines, desc='Processing', unit=' lines')
for fpath in jsonl_files:
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            pbar.update(1)
            post = json.loads(line)
            if post.get('analysis_result', {}).get('is_false_positive'):
                skip_fp += 1; continue
            groups = post.get('analysis_result', {}).get('analysis_groups', [])
            if not groups:
                skip_ng += 1; continue

            subreddit = post['subreddit']
            post_count_per_team[subreddit] = post_count_per_team.get(subreddit, 0) + 1
            ts_raw = post.get('created_utc')
            try:
                ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)

            for g in groups:
                if not isinstance(g, dict):
                    continue
                canonical = g.get('player_name_canonical', '') or ''
                if isinstance(canonical, list):
                    canonical = canonical[0] if canonical else ''
                if not isinstance(canonical, str) or canonical.lower() in ('none', '', 'unknown', 'n/a'):
                    continue

                raw_pos = g.get('player_position', '')
                if isinstance(raw_pos, list):
                    raw_pos = raw_pos[0] if raw_pos else ''
                cat = pos_to_cat(raw_pos)
                if cat is None:
                    skip_cat += 1; continue

                sent = g.get('sentiment_score')
                if sent is None:
                    skip_sent += 1; continue

                records.append({
                    'subreddit': subreddit,
                    'team_label': TEAM_LABELS.get(subreddit, subreddit),
                    'is_club': subreddit not in LEAGUE_SUBS,
                    'player': canonical,
                    'category': cat,
                    'sentiment': sent,
                    'date': ts.date(),
                    'ts': ts,
                })
pbar.close()

# Map subreddit post counts to team labels
team_post_count = {}
for sub, cnt in post_count_per_team.items():
    team = TEAM_LABELS.get(sub, sub)
    team_post_count[team] = team_post_count.get(team, 0) + cnt

df = pd.DataFrame(records)
print(f'\n{"="*60}')
print(f'  Cleaned records : {len(df):,}')
print(f'  Clubs           : {df[df["is_club"]].shape[0]:,}')
print(f'  Leagues         : {df[~df["is_club"]].shape[0]:,}')
print(f'{"="*60}')
print(f'  Skipped: false_pos={skip_fp:,}  no_groups={skip_ng:,}  '
      f'no_cat={skip_cat:,}  no_sent={skip_sent:,}')
print(f'{"="*60}')
print(f'\nCategories:\n{df["category"].value_counts()}')
print(f'\nPer team/league:\n{df.groupby("team_label")["category"].count().sort_index()}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 5 — Aggregation (sentiment + mentions + mention-per-person)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  Aggregate: mean, std, count, 95% CI  +  mention-per-person
# ═══════════════════════════════════════════════════════════

CLUB_ORDER   = ['Barcelona', 'FC Bayern', 'Liverpool', 'Man United', 'Real Madrid']
LEAGUE_ORDER = ['Bundesliga', 'LaLiga', 'Ligue1', 'Premier League']
all_teams = CLUB_ORDER + [t for t in LEAGUE_ORDER if t in df['team_label'].unique()]

# ── Sentiment aggregation ──
stat = (df.groupby(['team_label', 'category'])['sentiment']
        .agg(['mean', 'std', 'count']).reset_index())
stat['ci95'] = 1.96 * stat['std'] / np.sqrt(stat['count'])

cat_order  = ['Attack', 'Defense', 'Goalkeeper', 'Midfielder']  # 4-cat
cat_order3 = ['Attack', 'Defense', 'Goalkeeper']                # 3-cat

# 4-category pivots
pivot_m  = stat.pivot_table(index='team_label', columns='category', values='mean'
               ).reindex(index=all_teams, columns=cat_order)
pivot_ci = stat.pivot_table(index='team_label', columns='category', values='ci95'
               ).reindex(index=all_teams, columns=cat_order)
pivot_c  = stat.pivot_table(index='team_label', columns='category', values='count'
               ).reindex(index=all_teams, columns=cat_order)

# 3-category pivots (no Midfielder)
pivot_m3  = stat.pivot_table(index='team_label', columns='category', values='mean'
                ).reindex(index=all_teams, columns=cat_order3)
pivot_ci3 = stat.pivot_table(index='team_label', columns='category', values='ci95'
                ).reindex(index=all_teams, columns=cat_order3)
pivot_c3  = stat.pivot_table(index='team_label', columns='category', values='count'
                ).reindex(index=all_teams, columns=cat_order3)

print("=== 4-CATEGORY ===")
print("Mean sentiment:\n", pivot_m.round(3))
print("\nCount:\n", pivot_c.astype('Int64'))
print("\n=== 3-CATEGORY (no Midfielder) ===")
print("Mean sentiment:\n", pivot_m3.round(3))
print("\nCount:\n", pivot_c3.astype('Int64'))

# ── Mention-per-person aggregation ──
mpp = (df.groupby(['team_label', 'category'])
       .agg(total_mentions=('player', 'count'), unique_players=('player', 'nunique'))
       .reset_index())
mpp['mpp'] = mpp['total_mentions'] / mpp['unique_players']

# 4-category
pivot_mpp   = mpp.pivot_table(index='team_label', columns='category', values='mpp'
                  ).reindex(index=all_teams, columns=cat_order)
pivot_mpp_u = mpp.pivot_table(index='team_label', columns='category', values='unique_players'
                  ).reindex(index=all_teams, columns=cat_order)
# 3-category
pivot_mpp3   = mpp.pivot_table(index='team_label', columns='category', values='mpp'
                   ).reindex(index=all_teams, columns=cat_order3)
pivot_mpp_u3 = mpp.pivot_table(index='team_label', columns='category', values='unique_players'
                   ).reindex(index=all_teams, columns=cat_order3)

# ── Mention-per-person-per-thousand-posts ──
for team in all_teams:
    n_posts = team_post_count.get(team, 1)
    for cat in cat_order:
        if team in pivot_mpp.index and cat in pivot_mpp.columns and pd.notna(pivot_mpp.loc[team, cat]):
            pivot_mpp.loc[team, cat] = pivot_mpp.loc[team, cat] * 1000 / n_posts
    for cat in cat_order3:
        if team in pivot_mpp3.index and cat in pivot_mpp3.columns and pd.notna(pivot_mpp3.loc[team, cat]):
            pivot_mpp3.loc[team, cat] = pivot_mpp3.loc[team, cat] * 1000 / n_posts

print(f"\nPosts per team: {team_post_count}")
print("\n=== MENTION-PER-PERSON-PER-1K-POSTS (4-cat) ===")
print(pivot_mpp.round(2).to_string())
print("\n=== MENTION-PER-PERSON-PER-1K-POSTS (3-cat) ===")
print(pivot_mpp3.round(2).to_string())

# ── Per-team-per-category player mention counts ──
player_counts = df.groupby(['team_label', 'category', 'player']).size().reset_index(name='mentions')

# ── Top-10 per team per category (clubs only) ──
top10_all = (player_counts[player_counts['team_label'].isin(CLUB_ORDER)]
             .groupby(['team_label', 'category'])
             .apply(lambda g: g.nlargest(10, 'mentions')['mentions'].sum() / min(10, len(g)))
             .reset_index(name='avg_mentions_per_player'))
top10_all['mppk'] = top10_all.apply(
    lambda r: r['avg_mentions_per_player'] * 1000 / team_post_count.get(r['team_label'], 1), axis=1)

pivot_top10 = top10_all.pivot_table(index='team_label', columns='category', values='mppk'
                                    ).reindex(index=CLUB_ORDER, columns=cat_order)
pivot_top10_3 = top10_all.pivot_table(index='team_label', columns='category', values='mppk'
                                      ).reindex(index=CLUB_ORDER, columns=cat_order3)

# ── ≥5 mentions filter (all teams) ──
freq5_all = (player_counts[player_counts['mentions'] >= 5]
             .groupby(['team_label', 'category'])
             .apply(lambda g: g['mentions'].sum() / len(g))
             .reset_index(name='avg_mentions_per_player'))
freq5_all['mppk'] = freq5_all.apply(
    lambda r: r['avg_mentions_per_player'] * 1000 / team_post_count.get(r['team_label'], 1), axis=1)

pivot_freq5 = freq5_all.pivot_table(index='team_label', columns='category', values='mppk'
                                    ).reindex(index=all_teams, columns=cat_order)
pivot_freq5_3 = freq5_all.pivot_table(index='team_label', columns='category', values='mppk'
                                      ).reindex(index=all_teams, columns=cat_order3)

print("\n=== TOP-10 MENTIONS PER PLAYER PER 1K POSTS (clubs only, 4-cat) ===")
print(pivot_top10.round(2).to_string())
print("\n=== TOP-10 MENTIONS PER PLAYER PER 1K POSTS (clubs only, 3-cat) ===")
print(pivot_top10_3.round(2).to_string())
print("\n=== ≥5 MENTIONS PER PLAYER PER 1K POSTS (all teams, 4-cat) ===")
print(pivot_freq5.round(2).to_string())
print("\n=== ≥5 MENTIONS PER PLAYER PER 1K POSTS (all teams, 3-cat) ===")
print(pivot_freq5_3.round(2).to_string())
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 6 — Bar chart: sentiment mean (4 categories, wider gaps)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART 1 — Mean Sentiment (with 95% CI)
# ═══════════════════════════════════════════════════════════

COL = {'Attack': '#bc272d', 'Midfielder': '#e9c716',
       'Defense': '#0000a2', 'Goalkeeper': '#50ad9f'}
BW, GAP = 0.17, 0.35

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order):
        x = cur_x + ci * (BW + 0.03)
        val = pivot_m.loc[team, cat] if pd.notna(pivot_m.loc[team, cat]) else 0
        err = pivot_ci.loc[team, cat] if pd.notna(pivot_ci.loc[team, cat]) else 0
        ax.bar(x, val, BW, color=COL[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, yerr=err, capsize=3, error_kw={'linewidth': 1.2},
               label=cat if ti == 0 else '')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW + GAP

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.axhline(y=0, color='grey', linewidth=0.7, linestyle='--')
ax.set_ylabel('Mean Sentiment Score')
ax.set_title('Mean Sentiment by Player Category per Subreddit (with 95% CI)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'sentiment_mean_by_team.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "sentiment_mean_by_team.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 7 — Bar chart: mention count (4 categories, wider gaps)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART 2 — Mention Count
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_c = pivot_c.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order):
        x = cur_x + ci * (BW + 0.03)
        val = pivot_c.loc[team, cat] if pd.notna(pivot_c.loc[team, cat]) else 0
        ax.bar(x, val, BW, color=COL[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_c * 0.012, f'{int(val)}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW + GAP

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Mention Count')
ax.set_title('Mention Count by Player Category per Subreddit', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_count_by_team.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_count_by_team.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 8 — Bar chart: sentiment mean (3 categories, no Midfielder)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Mean Sentiment (3 categories, no Midfielder)
# ═══════════════════════════════════════════════════════════

BW3, GAP3 = 0.25, 0.40
COL3 = {'Attack': '#bc272d', 'Defense': '#0000a2', 'Goalkeeper': '#50ad9f'}

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order3):
        x = cur_x + ci * (BW3 + 0.04)
        val = pivot_m3.loc[team, cat] if pd.notna(pivot_m3.loc[team, cat]) else 0
        err = pivot_ci3.loc[team, cat] if pd.notna(pivot_ci3.loc[team, cat]) else 0
        ax.bar(x, val, BW3, color=COL3[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, yerr=err, capsize=3, error_kw={'linewidth': 1.2},
               label=cat if ti == 0 else '')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW3 + GAP3

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.axhline(y=0, color='grey', linewidth=0.7, linestyle='--')
ax.set_ylabel('Mean Sentiment Score')
ax.set_title('Mean Sentiment by Player Category per Subreddit (with 95% CI)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'sentiment_mean_by_team_3cat.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "sentiment_mean_by_team_3cat.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 9 — Bar chart: mention count (3 categories, no Midfielder)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Mention Count (3 categories, no Midfielder)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_c3 = pivot_c3.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order3):
        x = cur_x + ci * (BW3 + 0.04)
        val = pivot_c3.loc[team, cat] if pd.notna(pivot_c3.loc[team, cat]) else 0
        ax.bar(x, val, BW3, color=COL3[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_c3 * 0.012, f'{int(val)}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW3 + GAP3

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Mention Count')
ax.set_title('Mention Count by Player Category per Subreddit', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_count_by_team_3cat.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_count_by_team_3cat.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 10 — Bar chart: mention-per-person (4 categories)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Mentions per Person (4 categories)
#  total_mentions / unique_players per category per team
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_mpp = pivot_mpp.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order):
        x = cur_x + ci * (BW + 0.03)
        val = pivot_mpp.loc[team, cat] if pd.notna(pivot_mpp.loc[team, cat]) else 0
        ax.bar(x, val, BW, color=COL[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            n = int(pivot_mpp_u.loc[team, cat]) if pd.notna(pivot_mpp_u.loc[team, cat]) else 0
            ax.text(x, val + max_mpp * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW + GAP

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Mentions per Person per 1k Posts')
ax.set_title('Mentions per Person per 1k Posts by Player Category per Subreddit', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 11 — Bar chart: mention-per-person (3 categories, no Midfielder)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Mentions per Person (3 categories, no Midfielder)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_mpp3 = pivot_mpp3.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order3):
        x = cur_x + ci * (BW3 + 0.04)
        val = pivot_mpp3.loc[team, cat] if pd.notna(pivot_mpp3.loc[team, cat]) else 0
        ax.bar(x, val, BW3, color=COL3[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            n = int(pivot_mpp_u3.loc[team, cat]) if pd.notna(pivot_mpp_u3.loc[team, cat]) else 0
            ax.text(x, val + max_mpp3 * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW3 + GAP3

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Mentions per Person per 1k Posts')
ax.set_title('Mentions per Person per 1k Posts by Player Category per Subreddit', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person_3cat.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person_3cat.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 12 — Bar chart: top-10 mentions per player (clubs only, 4-cat)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Top-10 Mentions per Player (clubs only, 4-cat)
#  (sum of top-10 mentions / 10) * 1000 / n_posts
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_val = pivot_top10.max().max()
for ti, team in enumerate(CLUB_ORDER):
    offs = []
    for ci, cat in enumerate(cat_order):
        x = cur_x + ci * (BW + 0.03)
        val = pivot_top10.loc[team, cat] if pd.notna(pivot_top10.loc[team, cat]) else 0
        ax.bar(x, val, BW, color=COL[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_val * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW + GAP

ax.set_xticks(centers)
ax.set_xticklabels(CLUB_ORDER, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Avg Mentions per Top-10 Player per 1k Posts')
ax.set_title('Top-10 Player Mention Rate by Category (Clubs, per 1k Posts)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person_top10_club.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person_top10_club.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 13 — Bar chart: top-10 mentions per player (clubs only, 3-cat)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — Top-10 Mentions per Player (clubs only, 3-cat, no Midfielder)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_val = pivot_top10_3.max().max()
for ti, team in enumerate(CLUB_ORDER):
    offs = []
    for ci, cat in enumerate(cat_order3):
        x = cur_x + ci * (BW3 + 0.04)
        val = pivot_top10_3.loc[team, cat] if pd.notna(pivot_top10_3.loc[team, cat]) else 0
        ax.bar(x, val, BW3, color=COL3[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_val * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW3 + GAP3

ax.set_xticks(centers)
ax.set_xticklabels(CLUB_ORDER, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Avg Mentions per Top-10 Player per 1k Posts')
ax.set_title('Top-10 Player Mention Rate by Category (Clubs, per 1k Posts)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person_top10_club_3cat.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person_top10_club_3cat.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 14 — Bar chart: ≥5 mentions per player (all teams, 4-cat)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — ≥5 Mentions per Player (all teams, 4-cat)
#  (sum of mentions for players with >=5 mentions / n_freq) * 1000 / n_posts
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_val = pivot_freq5.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order):
        x = cur_x + ci * (BW + 0.03)
        val = pivot_freq5.loc[team, cat] if pd.notna(pivot_freq5.loc[team, cat]) else 0
        ax.bar(x, val, BW, color=COL[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_val * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW + GAP

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Avg Mentions per Frequent Player per 1k Posts')
ax.set_title('≥5 Mention Rate by Player Category per Subreddit (per 1k Posts)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person_freq5.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person_freq5.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 15 — Bar chart: ≥5 mentions per player (all teams, 3-cat)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART — ≥5 Mentions per Player (all teams, 3-cat, no Midfielder)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
cur_x = 0
centers = []
max_val = pivot_freq5_3.max().max()
for ti, team in enumerate(all_teams):
    offs = []
    for ci, cat in enumerate(cat_order3):
        x = cur_x + ci * (BW3 + 0.04)
        val = pivot_freq5_3.loc[team, cat] if pd.notna(pivot_freq5_3.loc[team, cat]) else 0
        ax.bar(x, val, BW3, color=COL3[cat], alpha=0.85, edgecolor='white',
               linewidth=0.5, label=cat if ti == 0 else '')
        if val > 0:
            ax.text(x, val + max_val * 0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        offs.append(x)
    centers.append((offs[0] + offs[-1]) / 2)
    cur_x = offs[-1] + BW3 + GAP3

ax.set_xticks(centers)
ax.set_xticklabels(all_teams, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Avg Mentions per Frequent Player per 1k Posts')
ax.set_title('≥5 Mention Rate by Player Category per Subreddit (per 1k Posts)', fontweight='bold')
ax.legend(title='Category', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'mention_per_person_freq5_3cat.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "mention_per_person_freq5_3cat.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 16 — Statistical testing
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  ANOVA + Pairwise t-tests (Bonferroni)
#  — all 4 categories: Attack, Defense, Goalkeeper, Midfielder
# ═══════════════════════════════════════════════════════════

print("=" * 78)
print("  ONE-WAY ANOVA  (H0: all category means equal)")
print("=" * 78)
for team in all_teams:
    sub = df[df['team_label'] == team]
    grps = {cat: sub[sub['category'] == cat]['sentiment'].values for cat in cat_order}
    valid = {k: v for k, v in grps.items() if len(v) > 1}
    if len(valid) < 2:
        print(f'  {team:20s}  insufficient data')
        continue
    f_stat, p_val = f_oneway(*list(valid.values()))
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    sizes = ', '.join(f'{k}={len(v)}' for k, v in valid.items())
    print(f'  {team:20s}  F={f_stat:.3f}  p={p_val:.2e}  {sig}  n=({sizes})')
print()

print("=" * 78)
print("  PAIRWISE T-TESTS (Welch, Bonferroni corrected)")
print("=" * 78)
for team in all_teams:
    sub = df[df['team_label'] == team]
    grps = {cat: sub[sub['category'] == cat]['sentiment'].values for cat in cat_order}
    valid = {k: v for k, v in grps.items() if len(v) > 1}
    if len(valid) < 2:
        continue
    pairs = list(combinations(sorted(valid.keys()), 2))
    nC = len(pairs)
    for c1, c2 in pairs:
        t_stat, p_raw = ttest_ind(valid[c1], valid[c2], equal_var=False)
        p_bonf = min(p_raw * nC, 1.0)
        sig = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else 'ns'
        print(f'  {team:20s}  {c1:12s} vs {c2:12s}:  t={t_stat:.3f}  '
              f'p_raw={p_raw:.2e}  p_bonf={p_bonf:.2e}  {sig}  '
              f'(n1={len(valid[c1])}, n2={len(valid[c2])})')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 17 — Time Series (clubs only)
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  CHART 3 — Club Sentiment Over Time
#  (no leagues; all clubs share same time axis)
# ═══════════════════════════════════════════════════════════

club_df = df[df['is_club']].copy()
print(f'Club records: {len(club_df):,}')
print('Clubs:', sorted(club_df['team_label'].unique()))

first_dates = club_df.groupby('team_label')['date'].min()
global_start = first_dates.max()
print(f'\nEarliest per club:\n{first_dates}')
print(f'\nGlobal X start: {global_start}')

club_df = club_df[club_df['date'] >= global_start].copy()

ts_daily = (club_df.groupby(['team_label', 'date'])['sentiment']
            .mean().reset_index())
ts_daily['date'] = pd.to_datetime(ts_daily['date'])

CLUB_PAL = {
    'Barcelona': '#a50044', 'FC Bayern': '#dc052d',
    'Liverpool': '#c8102e', 'Man United': '#da291c',
    'Real Madrid': '#febe10',
}

fig, ax = plt.subplots(figsize=(14, 5.5))
for club in CLUB_ORDER:
    sub = ts_daily[ts_daily['team_label'] == club].sort_values('date')
    if sub.empty:
        continue
    c = CLUB_PAL.get(club, '#333')
    ax.plot(sub['date'], sub['sentiment'], color=c, lw=0.6, alpha=0.4, label=None)
    sm = sub.copy()
    sm['s'] = sm['sentiment'].rolling(7, min_periods=3).mean()
    ax.plot(sm['date'], sm['s'], color=c, lw=2.2, alpha=0.85, label=club)

ax.axhline(y=0, color='grey', lw=0.7, ls='--')
ax.set_ylabel('Mean Daily Sentiment')
ax.set_title('Club Sentiment Over Time (daily mean + 7-day rolling smooth)', fontweight='bold')
ax.legend(fontsize=9, framealpha=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'sentiment_timeseries_clubs.png', dpi=200)
plt.show()
print(f'Saved -> {OUT_DIR / "sentiment_timeseries_clubs.png"}')
""")

# ══════════════════════════════════════════════════════════════════════
#  CELL 18 — Summary
# ══════════════════════════════════════════════════════════════════════
code(r"""# ═══════════════════════════════════════════════════════════
#  Summary Tables
# ═══════════════════════════════════════════════════════════

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 130)

print("=== MEAN SENTIMENT (with 95% CI) ===\n")
disp = pivot_m.round(3).copy()
for cat in cat_order:
    disp[f'{cat}  CI'] = pivot_ci[cat].round(3)
print(disp.to_string())
print()

print("=== MENTION COUNT ===\n")
print(pivot_c.astype('Int64').to_string())
print()

print(f"=== DATE RANGE ===")
print(f"  Overall  : {df['date'].min()}  ->  {df['date'].max()}")
print(f"  Club only: {global_start}  ->  {club_df['date'].max()}")
print(f"\nCharts saved to: {OUT_DIR}/")
""")

# ──── Write ──────────────────────────────────────────────────────────
nb.cells = cells
out = f'{BASE}/Computation/data_cleaning_and_visualisation.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook created -> {out}')
