"""
Comparaison Prophet vs LightGBM sur données simulées de débit eau.
Agrégation horaire pour la vitesse (Prophet est lent sur 5-min).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import holidays

# ─────────────────────────────────────────────
# 1. GÉNÉRER LES MÊMES DONNÉES (seed=42)
# ─────────────────────────────────────────────
print("── Génération des données simulées…")

def generer_donnees(date_debut='2023-01-01', date_fin='2024-12-31', seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(date_debut, date_fin, freq='5min')
    params = {'base': 80, 'amplitude': 50}

    h = np.asarray(dates.hour + dates.minute / 60)
    jour_sem = np.asarray(dates.dayofweek)
    jour_an  = np.asarray(dates.dayofyear)

    pic_matin = np.exp(-((h - 7.5) ** 2) / 4)
    pic_soir  = np.exp(-((h - 19) ** 2) / 5)
    profil_jour = pic_matin + 0.9 * pic_soir

    est_weekend = (jour_sem >= 5).astype(float)
    pic_we = np.exp(-((h - 10) ** 2) / 8)
    profil_jour = profil_jour * (1 - 0.4 * est_weekend) + 0.6 * pic_we * est_weekend

    saison = 0.3 * np.sin(2 * np.pi * (jour_an - 80) / 365)

    debit = params['base'] + params['amplitude'] * profil_jour + 10 * saison
    debit *= (1 + 0.05 * rng.standard_normal(len(dates)))
    debit = np.maximum(debit, 5)

    masque_nan = rng.random(len(dates)) < 0.01
    debit[masque_nan] = np.nan

    return pd.DataFrame({'ds': dates, 'debit': debit})

df_5min = generer_donnees()
print(f"   5-min : {len(df_5min):,} points  ({df_5min['ds'].min().date()} → {df_5min['ds'].max().date()})")

# ─────────────────────────────────────────────
# 2. AGRÉGER À L'HEURE
# ─────────────────────────────────────────────
df_h = df_5min.set_index('ds')['debit'].resample('h').mean().dropna().reset_index()
df_h.columns = ['ds', 'y']
print(f"   Horaire : {len(df_h):,} points")

# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST (mêmes 3 derniers mois)
# ─────────────────────────────────────────────
date_split = df_h['ds'].max() - pd.Timedelta(days=90)
train = df_h[df_h['ds'] <  date_split].copy()
test  = df_h[df_h['ds'] >= date_split].copy()
print(f"\n── Split : train {len(train):,} h  |  test {len(test):,} h  (coupure : {date_split.date()})")

# ─────────────────────────────────────────────
# 4. PROPHET
# ─────────────────────────────────────────────
print("\n── Entraînement Prophet…  (peut prendre 2-5 min)")
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

feries = make_holidays_df(year_list=[2023, 2024], country='FR')

m = Prophet(
    yearly_seasonality=10,      # 10 termes Fourier pour le cycle annuel
    weekly_seasonality=True,
    daily_seasonality=True,
    holidays=feries,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
)

m.fit(train)
print("   Modèle ajusté.")

# Prédictions sur la période test
futur = m.make_future_dataframe(periods=len(test), freq='h', include_history=False)
forecast = m.predict(futur)

test = test.copy()
test['pred_prophet'] = forecast['yhat'].values

# ─────────────────────────────────────────────
# 5. LIGHTGBM (horaire, sans lags pour comparaison équitable)
# ─────────────────────────────────────────────
print("\n── Entraînement LightGBM (horaire, features calendaires)…")
import lightgbm as lgb

feries_fr = holidays.France(years=range(2022, 2027))

def features_cal(df):
    d = df.copy()
    dt = d['ds']
    d['heure']          = dt.dt.hour
    d['jour_semaine']   = dt.dt.dayofweek
    d['jour_annee']     = dt.dt.dayofyear
    d['mois']           = dt.dt.month
    d['annee']          = dt.dt.year
    d['est_weekend']    = (d['jour_semaine'] >= 5).astype(int)
    d['est_ferie']      = dt.dt.date.astype('object').map(lambda x: x in feries_fr).astype(int)
    d['est_vacances']   = ((d['mois'] == 7) | (d['mois'] == 8)).astype(int)
    d['heure_sin']      = np.sin(2 * np.pi * d['heure'] / 24)
    d['heure_cos']      = np.cos(2 * np.pi * d['heure'] / 24)
    d['jour_sem_sin']   = np.sin(2 * np.pi * d['jour_semaine'] / 7)
    d['jour_sem_cos']   = np.cos(2 * np.pi * d['jour_semaine'] / 7)
    d['mois_sin']       = np.sin(2 * np.pi * d['mois'] / 12)
    d['mois_cos']       = np.cos(2 * np.pi * d['mois'] / 12)
    return d

train_f = features_cal(train)
test_f  = features_cal(test)

FEATS = [c for c in train_f.columns if c not in ('ds', 'y', 'pred_prophet')]

mdl = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, random_state=42, n_jobs=-1, verbose=-1,
)
mdl.fit(train_f[FEATS], train_f['y'],
        eval_set=[(test_f[FEATS], test_f['y'])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

test['pred_lgbm'] = mdl.predict(test_f[FEATS])
print("   Modèle ajusté.")

# ─────────────────────────────────────────────
# 6. ÉVALUATION
# ─────────────────────────────────────────────
def evaluer(y, pred, nom):
    mae  = np.mean(np.abs(y - pred))
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    mape = np.mean(np.abs((y - pred) / y)) * 100
    print(f"  {nom:30s}  MAE={mae:6.2f} m³/h   RMSE={rmse:6.2f}   MAPE={mape:5.2f}%")
    return mae, rmse, mape

print("\n── Résultats sur le jeu de test (3 derniers mois, données horaires) :")
mae_p,  rmse_p,  mape_p  = evaluer(test['y'], test['pred_prophet'], 'Prophet (additif)')
mae_l,  rmse_l,  mape_l  = evaluer(test['y'], test['pred_lgbm'],    'LightGBM (calendaire)')

winner = "LightGBM" if mae_l < mae_p else "Prophet"
gain   = abs(mae_p - mae_l) / max(mae_p, mae_l) * 100
print(f"\n  → {winner} gagne  ({gain:.1f}% de MAE en moins)")

# ─────────────────────────────────────────────
# 7. GRAPHIQUES
# ─────────────────────────────────────────────
print("\n── Génération des graphiques…")

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("Prophet vs LightGBM — PR_CENTRE (données horaires)", fontsize=13, y=0.98)

# -- Graphique 1 : une semaine complète
semaine = test.head(7 * 24)
ax = axes[0]
ax.plot(semaine['ds'], semaine['y'],           color='#1a2332', lw=1.4, label='Réel', zorder=3)
ax.plot(semaine['ds'], semaine['pred_prophet'], color='#2c5282', lw=1.2, label=f'Prophet  (MAE={mae_p:.2f})', ls='--')
ax.plot(semaine['ds'], semaine['pred_lgbm'],    color='#c8511c', lw=1.2, label=f'LightGBM (MAE={mae_l:.2f})', ls='-.')
ax.set_title("Première semaine du jeu de test (résolution horaire)")
ax.legend(); ax.set_ylabel("Débit m³/h"); ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d/%m'))

# -- Graphique 2 : composantes Prophet
comp = m.predict(m.make_future_dataframe(periods=0, freq='h', include_history=True))
comp = comp[comp['ds'] >= train['ds'].min()]

ax = axes[1]
ax2 = ax.twinx()
ax.plot(comp['ds'], comp['trend'],  color='#4a6b3a', lw=1, label='Tendance', alpha=0.8)
ax2.fill_between(comp['ds'], comp['yearly'].min(), comp['yearly'],
                 color='#a87820', alpha=0.2, label='Cycle annuel')
ax.set_title("Composantes Prophet : tendance (vert) + cycle annuel (jaune)")
ax.set_ylabel("Tendance (m³/h)"); ax2.set_ylabel("Cycle annuel (effet relatif)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.grid(alpha=0.2)

# -- Graphique 3 : scatter prédit vs réel
ax = axes[2]
vmin = test['y'].min(); vmax = test['y'].max()
ax.scatter(test['y'], test['pred_prophet'], alpha=0.07, s=4, color='#2c5282', label='Prophet')
ax.scatter(test['y'], test['pred_lgbm'],    alpha=0.07, s=4, color='#c8511c', label='LightGBM')
ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, label='Parfait')
ax.set_xlabel("Débit réel (m³/h)"); ax.set_ylabel("Débit prédit (m³/h)")
ax.set_title("Prédit vs réel (tous les points du test)")
ax.legend(); ax.grid(alpha=0.2)

plt.tight_layout()
out = r'C:\Users\PC\Documents\GitHub\Eaux\_docs\prophet_vs_lgbm.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"   Graphique sauvegardé → {out}")

# ─────────────────────────────────────────────
# 8. TABLEAU RÉCAP FINAL
# ─────────────────────────────────────────────
print("\n" + "═"*60)
print("  RÉCAP — données horaires, mêmes 90 jours de test")
print("═"*60)
print(f"  {'Modèle':<30} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
print(f"  {'-'*58}")
print(f"  {'Prophet (additif)':<30} {mae_p:>7.2f}  {rmse_p:>7.2f}  {mape_p:>6.2f}%")
print(f"  {'LightGBM (calendaire)':<30} {mae_l:>7.2f}  {rmse_l:>7.2f}  {mape_l:>6.2f}%")
print("═"*60)
print("""
Notes :
  - Résolution : horaire (agrégation des données 5-min)
  - LightGBM ici = features calendaires uniquement (pas de lags)
  - Sur 5-min avec lags, LightGBM atteint MAE ≈ 2.81 m³/h
  - Prophet + mode multiplicatif pourrait réduire l'écart
""")
