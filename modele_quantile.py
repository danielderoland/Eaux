"""
LightGBM avec régression quantile — intervalles de confiance pour la régulation des PR.
Données 5-min avec lags (court terme, opérationnel).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
import holidays
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DONNÉES (même seed que le notebook)
# ─────────────────────────────────────────────
print("── Génération des données 5-min…")

def generer(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='5min')
    h   = dates.hour + dates.minute / 60
    jsem = dates.dayofweek
    jan  = dates.dayofyear

    pic_m = np.exp(-((h - 7.5)**2) / 4)
    pic_s = np.exp(-((h - 19 )**2) / 5)
    profil = pic_m + 0.9 * pic_s
    we = (jsem >= 5).astype(float)
    pic_we = np.exp(-((h - 10)**2) / 8)
    profil = profil * (1 - 0.4*we) + 0.6 * pic_we * we
    saison = 0.3 * np.sin(2*np.pi*(jan - 80)/365)

    debit = 80 + 50*profil + 10*saison
    debit *= (1 + 0.05 * rng.standard_normal(len(dates)))
    debit  = np.maximum(debit, 5)
    mask = rng.random(len(dates)) < 0.01
    debit = np.where(mask, np.nan, debit)

    return pd.DataFrame({'datetime': np.asarray(dates), 'debit': debit})

df = generer()
print(f"   {len(df):,} points  ({df['datetime'].min().date()} → {df['datetime'].max().date()})")

# ─────────────────────────────────────────────
# 2. FEATURES CALENDAIRES + LAGS
# ─────────────────────────────────────────────
print("── Feature engineering…")

feries_fr = holidays.France(years=range(2022, 2027))

def build_features(df):
    d = df.sort_values('datetime').copy()
    dt = d['datetime']

    d['heure']        = dt.dt.hour
    d['minute']       = dt.dt.minute
    d['jour_semaine'] = dt.dt.dayofweek
    d['jour_annee']   = dt.dt.dayofyear
    d['mois']         = dt.dt.month
    d['annee']        = dt.dt.year
    d['est_weekend']  = (d['jour_semaine'] >= 5).astype(int)
    d['est_ferie']    = dt.dt.date.astype('object').map(lambda x: x in feries_fr).astype(int)
    d['est_vacances'] = ((d['mois'] == 7) | (d['mois'] == 8)).astype(int)

    d['heure_sin']     = np.sin(2*np.pi * (d['heure'] + d['minute']/60) / 24)
    d['heure_cos']     = np.cos(2*np.pi * (d['heure'] + d['minute']/60) / 24)
    d['jour_sem_sin']  = np.sin(2*np.pi * d['jour_semaine'] / 7)
    d['jour_sem_cos']  = np.cos(2*np.pi * d['jour_semaine'] / 7)
    d['mois_sin']      = np.sin(2*np.pi * d['mois'] / 12)
    d['mois_cos']      = np.cos(2*np.pi * d['mois'] / 12)

    # Lags : 5 min, 15 min, 1h, 6h, 24h, 7 jours
    for lag in [1, 3, 12, 72, 288, 2016]:
        d[f'lag_{lag}'] = d['debit'].shift(lag)

    # Moyennes mobiles 1h et 24h (décalées d'un pas)
    for w in [12, 288]:
        roll = d['debit'].shift(1).rolling(w, min_periods=1)
        d[f'roll_mean_{w}'] = roll.mean()
        d[f'roll_std_{w}']  = roll.std()

    return d

df = build_features(df)

# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
date_split = df['datetime'].max() - pd.Timedelta(days=90)
train = df[df['datetime'] <  date_split].dropna(subset=['debit'])
test  = df[df['datetime'] >= date_split].dropna(subset=['debit'])

FEATS = [c for c in train.columns if c not in ('datetime', 'debit')]

print(f"   Train : {len(train):,}  |  Test : {len(test):,}")

# ─────────────────────────────────────────────
# 4. RÉGRESSION QUANTILE — 3 modèles
# ─────────────────────────────────────────────
print("\n── Entraînement des 3 modèles quantile (q=0.1 / 0.5 / 0.9)…")

PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    random_state=42, n_jobs=-1, verbose=-1,
)

preds = {}
for q in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, **PARAMS)
    m.fit(
        train[FEATS], train['debit'],
        eval_set=[(test[FEATS], test['debit'])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    preds[q] = m.predict(test[FEATS])
    print(f"   q={q} : OK  (best iter={m.best_iteration_})")

test = test.copy()
test['q10'] = preds[0.1]
test['q50'] = preds[0.5]
test['q90'] = preds[0.9]

# ─────────────────────────────────────────────
# 5. MÉTRIQUES
# ─────────────────────────────────────────────
mae   = np.mean(np.abs(test['debit'] - test['q50']))
mape  = np.mean(np.abs((test['debit'] - test['q50']) / test['debit'])) * 100
cover = np.mean((test['debit'] >= test['q10']) & (test['debit'] <= test['q90'])) * 100
width = np.mean(test['q90'] - test['q10'])

print(f"\n── Résultats (jeu de test, 90 derniers jours) :")
print(f"   Médiane (q50)  →  MAE = {mae:.2f} m³/h   MAPE = {mape:.2f}%")
print(f"   Intervalle [q10–q90]  →  couverture = {cover:.1f}%  (cible 80%)")
print(f"   Largeur moyenne de l'intervalle = {width:.1f} m³/h")

# ─────────────────────────────────────────────
# 6. GRAPHIQUES
# ─────────────────────────────────────────────
print("\n── Génération des graphiques…")

fig = plt.figure(figsize=(14, 16))
fig.patch.set_facecolor('#f5f1e8')

ORANGE = '#c8511c'
BLUE   = '#2c5282'
TEAL   = '#1d6b6b'
INK    = '#1a2332'
PAPER  = '#f5f1e8'

# ── Graphique 1 : semaine avec bande de confiance ──────────────────────────
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_facecolor(PAPER)

# Première semaine disponible dans le test (lundi → dimanche)
t0 = test[test['datetime'].dt.dayofweek == 0]['datetime'].iloc[0]
t0 = t0.replace(hour=0, minute=0)
t1 = t0 + pd.Timedelta(days=7)
wk = test[(test['datetime'] >= t0) & (test['datetime'] < t1)].copy()

ax1.fill_between(wk['datetime'], wk['q10'], wk['q90'],
                 color=ORANGE, alpha=0.15, label='Intervalle [q10–q90]')
ax1.plot(wk['datetime'], wk['debit'], color=INK,   lw=1.4, label='Réel',       zorder=4)
ax1.plot(wk['datetime'], wk['q50'],   color=ORANGE, lw=1.2, label='Médiane q50', zorder=3)
ax1.plot(wk['datetime'], wk['q10'],   color=ORANGE, lw=0.7, ls=':', alpha=0.7)
ax1.plot(wk['datetime'], wk['q90'],   color=ORANGE, lw=0.7, ls=':', alpha=0.7)

# Marquer les pics détectés (q50 > 95 m³/h en heure de pointe)
peaks = wk[(wk['q50'] > 95) & (wk['datetime'].dt.hour.between(6, 10) |
                                 wk['datetime'].dt.hour.between(17, 21))]
ax1.scatter(peaks['datetime'], peaks['q50'] + 4, marker='v',
            color=TEAL, s=30, zorder=5, label='Pic détecté → ouvrir PR')

ax1.set_title(f"Semaine du {t0.strftime('%d/%m/%Y')} — prédiction avec intervalle de confiance",
              fontsize=11, color=INK, pad=10)
ax1.legend(fontsize=9, framealpha=0.9)
ax1.set_ylabel("Débit (m³/h)")
ax1.grid(alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %d\n%Hh'))

# ── Graphique 2 : couverture — vrai vs intervalles ────────────────────────
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_facecolor(PAPER)

# Sous-échantillonner pour lisibilité (1 point / heure)
sub = test.iloc[::12].copy()
in_band  = (sub['debit'] >= sub['q10']) & (sub['debit'] <= sub['q90'])
out_band = ~in_band

ax2.fill_between(sub['datetime'], sub['q10'], sub['q90'],
                 color=ORANGE, alpha=0.12, label=f'Intervalle [q10–q90]  couverture={cover:.0f}%')
ax2.scatter(sub[in_band]['datetime'],  sub[in_band]['debit'],
            s=3, color=TEAL,   alpha=0.5, label='Dans l\'intervalle')
ax2.scatter(sub[out_band]['datetime'], sub[out_band]['debit'],
            s=8, color='#c0392b', alpha=0.8, zorder=4, label='Hors intervalle')

ax2.set_title(f"Couverture de l'intervalle sur les 90 jours de test  "
              f"(1 point/heure affiché  ·  largeur moy. = {width:.0f} m³/h)",
              fontsize=11, color=INK, pad=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_ylabel("Débit (m³/h)")
ax2.grid(alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# ── Graphique 3 : schéma décisionnel ──────────────────────────────────────
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_facecolor(PAPER)
ax3.axis('off')

# Simuler 6 heures autour du pic matin pour montrer la logique d'ouverture
h_range = np.linspace(5, 11, 200)
q50  = 80 + 50*(np.exp(-((h_range-7.5)**2)/4) + 0.9*np.exp(-((h_range-19)**2)/5))
q10  = q50 * 0.93
q90  = q50 * 1.08
seuil = 95  # m³/h → seuil d'ouverture du PR
delai = 0.33  # 20 min de délai hydraulique

ax_dec = fig.add_axes([0.08, 0.04, 0.84, 0.26])
ax_dec.set_facecolor(PAPER)
ax_dec.fill_between(h_range, q10, q90, color=ORANGE, alpha=0.15, label='Intervalle [q10–q90]')
ax_dec.plot(h_range, q50, color=ORANGE, lw=2, label='Prédiction médiane')
ax_dec.axhline(seuil, color=TEAL, lw=1.5, ls='--', label=f'Seuil ouverture = {seuil} m³/h')

# Trouver le moment où q50 dépasse le seuil
idx_seuil = np.argmax(q50 >= seuil)
t_depasse = h_range[idx_seuil]
t_ouvrir  = t_depasse - delai

ax_dec.axvline(t_depasse, color=TEAL,     lw=1, ls=':', alpha=0.7)
ax_dec.axvline(t_ouvrir,  color='#a87820', lw=2, label=f'→ Ouvrir PR à {t_ouvrir:.0f}h{int((t_ouvrir%1)*60):02d} (délai 20 min)')

ax_dec.annotate(f"Pic prédit\nà {t_depasse:.1f}h",
                xy=(t_depasse, seuil), xytext=(t_depasse+0.3, seuil+5),
                fontsize=9, color=TEAL,
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1))
ax_dec.annotate(f"Ouvrir maintenant\n(t={t_ouvrir:.0f}h{int((t_ouvrir%1)*60):02d})",
                xy=(t_ouvrir, 82), xytext=(t_ouvrir-1.2, 72),
                fontsize=9, color='#a87820',
                arrowprops=dict(arrowstyle='->', color='#a87820', lw=1))

ax_dec.set_xlim(5, 11)
ax_dec.set_xlabel("Heure")
ax_dec.set_ylabel("Débit (m³/h)")
ax_dec.set_title("Logique de régulation : ouvrir le PR avant que la demande dépasse le seuil",
                 fontsize=11, color=INK)
ax_dec.legend(fontsize=9, framealpha=0.9)
ax_dec.grid(alpha=0.2)
ax_dec.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{int(v)}h{int((v%1)*60):02d}"))

out = r'C:\Users\PC\Documents\GitHub\Eaux\_docs\quantile_regulation.png'
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=PAPER)
print(f"   Graphique sauvegardé → {out}")

# ─────────────────────────────────────────────
# 7. RÉCAP
# ─────────────────────────────────────────────
print("\n" + "═"*60)
print("  RÉGRESSION QUANTILE — Résumé opérationnel")
print("═"*60)
print(f"  Médiane (q50)   MAE  = {mae:.2f} m³/h  |  MAPE = {mape:.2f}%")
print(f"  Intervalle [q10–q90]  couverture = {cover:.1f}%  (cible 80%)")
print(f"  Largeur moy. = {width:.1f} m³/h")
print("""
  Lecture opérationnelle :
  · q50 = meilleure estimation du débit à venir
  · q10 = scénario bas  → PR fermé, pas de risque de débordement
  · q90 = scénario haut → PR ouvert en anticipe, sécurité maximale
  · Si q10 > seuil : ouvrir le PR MAINTENANT (pic quasi certain)
  · Si q90 < seuil : ne rien faire (pic improbable)
  · Si q10 < seuil < q90 : décision selon politique de risque
""")
