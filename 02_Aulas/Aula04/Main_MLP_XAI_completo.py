# %% ============================================================
# MLP + FEATURE SELECTION + XAI + INCERTEZA EPISTEMICA
# ===============================================================
# Objetivos:
# 1) Comparar MLP com todas as variaveis vs selecao por correlacao,
#    informacao mutua e SHAP.
# 2) Explicar o modelo com SHAP globalmente e por regime hidrologico.
# 3) Comparar importancias em baixa vazao, condicao normal e alta vazao.
# 4) Estimar incerteza epistemica via Monte Carlo Dropout.
# 5) Relacionar incerteza, erro e regime hidrologico.
#
# Requisitos:
# pip install pandas numpy matplotlib scikit-learn torch shap
# ===============================================================

from pathlib import Path
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

warnings.filterwarnings("ignore")

# %% ============================================================
# 1. CONFIGURACOES GERAIS
# ===============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

T = 1                  # Horizonte de previsao: Q_Afluente(t+T)
TOP_N = 6              # Numero de variaveis selecionadas por metodo
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

EPOCHS = 2000
BATCH_SIZE = 365
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 120
DROPOUT = 0.20

MC_SAMPLES = 300       # Monte Carlo Dropout
INTERVAL_ALPHA = 0.05  # intervalo 95%

LOW_Q = 0.20           # baixa vazao <= Q20 do treino
HIGH_Q = 0.80          # alta vazao >= Q80 do treino

SHAP_BACKGROUND_N = 200
SHAP_EXPLAIN_N = 600
LOCAL_EXAMPLES_PER_REGIME = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "series_preenchidas.csv"
OUT_DIR = BASE_DIR / "resultados_mlp_xai"
OUT_DIR.mkdir(exist_ok=True)

print(f"Dispositivo: {DEVICE}")
print(f"Arquivo: {ARQUIVO_DADOS}")
print(f"Saidas: {OUT_DIR}")

# %% ============================================================
# 2. FUNCOES AUXILIARES
# ===============================================================

def nse(y_true, y_pred):
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / den


def calc_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "NSE": nse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


class MLP(nn.Module):
    def __init__(self, input_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_train, y_train, X_val, y_val, label):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).view(-1, 1)
    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE).view(-1, 1)

    model = MLP(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(X_train_t.size(0), device=DEVICE)
        epoch_losses = []

        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"[{label}] Epoca {epoch+1}/{EPOCHS} | treino={train_loss:.6f} | val={val_loss:.6f}")

        if patience_counter >= PATIENCE:
            print(f"[{label}] Early stopping na epoca {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, scaler, train_losses, val_losses


def deterministic_predict(model, scaler, X):
    X_sc = scaler.transform(X)
    X_t = torch.tensor(X_sc, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).detach().cpu().numpy().flatten()
    return pred


def mc_dropout_predict(model, scaler, X, n_samples=MC_SAMPLES):
    X_sc = scaler.transform(X)
    X_t = torch.tensor(X_sc, dtype=torch.float32, device=DEVICE)

    preds = []
    model.train()  # ativa dropout
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(X_t).detach().cpu().numpy().flatten()
            preds.append(p)

    model.eval()
    preds = np.asarray(preds)

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    lower = np.quantile(preds, INTERVAL_ALPHA / 2, axis=0)
    upper = np.quantile(preds, 1 - INTERVAL_ALPHA / 2, axis=0)
    return mean, std, lower, upper


def regime_from_y(y, q_low, q_high):
    r = np.full(len(y), "Normal", dtype=object)
    r[y <= q_low] = "Baixa vazao"
    r[y >= q_high] = "Alta vazao"
    return r


def save_loss_plot(histories):
    plt.figure(figsize=(10, 5))
    for label, (tr, va) in histories.items():
        plt.plot(va, label=f"{label} - validacao", linewidth=1.3)
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "convergencia_modelos.png", dpi=200)
    plt.close()


def shap_values_for_model(model, scaler, X_train, X_explain, feature_names):
    # KernelExplainer e agnostico ao framework e tende a ser mais robusto
    rng = np.random.default_rng(SEED)

    n_bg = min(SHAP_BACKGROUND_N, len(X_train))
    n_ex = min(SHAP_EXPLAIN_N, len(X_explain))

    bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
    ex_idx = rng.choice(len(X_explain), size=n_ex, replace=False)

    background = X_train[bg_idx]
    X_exp = X_explain[ex_idx]

    def predict_fn(X_np):
        return deterministic_predict(model, scaler, np.asarray(X_np)).reshape(-1)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(X_exp, nsamples="auto")
    shap_vals = np.asarray(shap_vals)

    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., 0]

    return explainer, X_exp, shap_vals, ex_idx


def global_shap_importance(shap_vals, feature_names):
    imp = np.mean(np.abs(shap_vals), axis=0)
    df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
    return df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_shap_bar(df_imp, filename, title=None):
    d = df_imp.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, max(4, 0.35 * len(d))))
    plt.barh(d["feature"], d["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def plot_regime_importance(regime_imp):
    pivot = regime_imp.pivot(index="feature", columns="regime", values="mean_abs_shap").fillna(0)
    score = pivot.max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[score.index]
    pivot = pivot.head(min(12, len(pivot)))

    x = np.arange(len(pivot))
    width = 0.25
    regimes = [c for c in ["Baixa vazao", "Normal", "Alta vazao"] if c in pivot.columns]

    plt.figure(figsize=(12, 6))
    offsets = np.linspace(-width, width, len(regimes)) if len(regimes) > 1 else [0]
    for off, reg in zip(offsets, regimes):
        plt.bar(x + off, pivot[reg].values, width=width, label=reg)

    plt.xticks(x, pivot.index, rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_importancia_por_regime.png", dpi=200)
    plt.close()


# %% ============================================================
# 3. CARREGAMENTO DOS DADOS
# ===============================================================

df = pd.read_csv(ARQUIVO_DADOS)

if "data" in df.columns:
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
else:
    df["data"] = pd.RangeIndex(len(df))

if "Q_Afluente" not in df.columns:
    raise ValueError("A coluna 'Q_Afluente' nao foi encontrada.")

df["target"] = df["Q_Afluente"].shift(-T)
df["data_target"] = df["data"].shift(-T)
df = df.dropna().reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ["data", "data_target", "Q_Afluente", "target"]]

X_raw = df[feature_cols].copy()
y = df["target"].to_numpy()
dates = df["data_target"].to_numpy()

print(f"Registros: {len(df)}")
print(f"Variaveis candidatas: {len(feature_cols)}")

# %% ============================================================
# 4. SPLIT TEMPORAL: TREINO / VALIDACAO / TESTE
# ===============================================================

n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

X_train_all = X_raw.iloc[:train_end].copy()
X_val_all = X_raw.iloc[train_end:val_end].copy()
X_test_all = X_raw.iloc[val_end:].copy()

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

dates_test = dates[val_end:]

# %% ============================================================
# 5. FEATURE SELECTION APENAS NO TREINO
# ===============================================================

# Correlação absoluta
train_corr = pd.concat(
    [X_train_all.reset_index(drop=True), pd.Series(y_train, name="target")], axis=1
).corr(numeric_only=True)

corr_series = train_corr["target"].drop("target")
corr_rank = corr_series.abs().sort_values(ascending=False)
top_corr_vars = corr_rank.head(TOP_N).index.tolist()

# Informacao mutua
mi_scores = mutual_info_regression(X_train_all, y_train, random_state=SEED)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
top_mi_vars = mi_series.head(TOP_N).index.tolist()

pd.DataFrame({
    "feature": corr_rank.index,
    "abs_correlation": corr_rank.values,
    "signed_correlation": corr_series.loc[corr_rank.index].values,
}).to_csv(OUT_DIR / "ranking_correlacao_treino.csv", index=False)

pd.DataFrame({
    "feature": mi_series.index,
    "mutual_information": mi_series.values,
}).to_csv(OUT_DIR / "ranking_informacao_mutua_treino.csv", index=False)

print("Top correlacao:", top_corr_vars)
print("Top MI:", top_mi_vars)

# %% ============================================================
# 6. TREINO MLP-ALL
# ===============================================================

models = {}
scalers = {}
histories = {}
metrics_rows = []

model_all, scaler_all, tr_all, va_all = train_model(
    X_train_all.values, y_train,
    X_val_all.values, y_val,
    "MLP-All"
)
models["MLP-All"] = model_all
scalers["MLP-All"] = scaler_all
histories["MLP-All"] = (tr_all, va_all)

pred_test_all = deterministic_predict(model_all, scaler_all, X_test_all.values)
met_all = calc_metrics(y_test, pred_test_all)
metrics_rows.append({"Modelo": "MLP-All", "N_features": len(feature_cols), **met_all})

# %% ============================================================
# 7. SHAP GLOBAL DO MLP-ALL E SELECAO TOP-N SHAP
# ===============================================================

print("Calculando SHAP do MLP-All...")
explainer_all, X_shap, shap_vals, shap_idx = shap_values_for_model(
    model_all,
    scaler_all,
    X_train_all.values,
    X_val_all.values,
    feature_cols,
)

shap_imp = global_shap_importance(shap_vals, feature_cols)
shap_imp.to_csv(OUT_DIR / "ranking_shap_global.csv", index=False)
plot_shap_bar(shap_imp, "shap_global_bar.png", "Importancia SHAP global - MLP-All")

top_shap_vars = shap_imp.head(TOP_N)["feature"].tolist()
print("Top SHAP:", top_shap_vars)

# Beeswarm
try:
    shap.summary_plot(
        shap_vals,
        pd.DataFrame(X_shap, columns=feature_cols),
        show=False,
        max_display=min(15, len(feature_cols)),
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_global_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Aviso: beeswarm SHAP nao gerado:", e)

# %% ============================================================
# 8. TREINO DOS MODELOS COM FEATURE SELECTION
# ===============================================================

selection_sets = {
    "MLP-Corr": top_corr_vars,
    "MLP-MI": top_mi_vars,
    "MLP-SHAP": top_shap_vars,
}

for label, cols in selection_sets.items():
    model, scaler, tr, va = train_model(
        X_train_all[cols].values,
        y_train,
        X_val_all[cols].values,
        y_val,
        label,
    )

    models[label] = model
    scalers[label] = scaler
    histories[label] = (tr, va)

    pred = deterministic_predict(model, scaler, X_test_all[cols].values)
    met = calc_metrics(y_test, pred)
    metrics_rows.append({"Modelo": label, "N_features": len(cols), **met})

# %% ============================================================
# 9. COMPARACAO DE DESEMPENHO
# ===============================================================

metrics_df = pd.DataFrame(metrics_rows).sort_values("NSE", ascending=False)
metrics_df.to_csv(OUT_DIR / "comparacao_modelos_feature_selection.csv", index=False)
print("\nComparacao final:")
print(metrics_df.round(4))

save_loss_plot(histories)

plt.figure(figsize=(9, 5))
plt.bar(metrics_df["Modelo"], metrics_df["NSE"])
plt.ylabel("NSE - teste")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "comparacao_nse_modelos.png", dpi=200)
plt.close()

# %% ============================================================
# 10. REGIMES HIDROLOGICOS DEFINIDOS PELO TREINO
# ===============================================================

q_low = np.quantile(y_train, LOW_Q)
q_high = np.quantile(y_train, HIGH_Q)

regime_test = regime_from_y(y_test, q_low, q_high)

pd.DataFrame({
    "limiar": ["Q_low", "Q_high"],
    "quantil": [LOW_Q, HIGH_Q],
    "valor": [q_low, q_high],
}).to_csv(OUT_DIR / "limiares_regime_hidrologico.csv", index=False)

# %% ============================================================
# 11. SHAP POR REGIME NO TESTE
# ===============================================================

# Para SHAP por regime usamos MLP-All, pois ele preserva todas as variaveis.
# Amostragem para controlar custo computacional.
rng = np.random.default_rng(SEED)
regime_rows = []
regime_shap_long = []

for regime_name in ["Baixa vazao", "Normal", "Alta vazao"]:
    mask = regime_test == regime_name
    X_reg = X_test_all.loc[mask].values

    if len(X_reg) == 0:
        continue

    n_reg = min(SHAP_EXPLAIN_N, len(X_reg))
    idx = rng.choice(len(X_reg), size=n_reg, replace=False)
    X_reg_sub = X_reg[idx]

    def predict_fn_reg(X_np):
        return deterministic_predict(model_all, scaler_all, np.asarray(X_np)).reshape(-1)

    # Reutiliza background do treino
    n_bg = min(SHAP_BACKGROUND_N, len(X_train_all))
    bg_idx = rng.choice(len(X_train_all), size=n_bg, replace=False)
    expl = shap.KernelExplainer(predict_fn_reg, X_train_all.values[bg_idx])
    sv = np.asarray(expl.shap_values(X_reg_sub, nsamples="auto"))
    if sv.ndim == 3:
        sv = sv[..., 0]

    imp = np.mean(np.abs(sv), axis=0)
    for f, v in zip(feature_cols, imp):
        regime_rows.append({"regime": regime_name, "feature": f, "mean_abs_shap": v})

    for i in range(len(X_reg_sub)):
        for j, f in enumerate(feature_cols):
            regime_shap_long.append({
                "regime": regime_name,
                "sample": i,
                "feature": f,
                "feature_value": X_reg_sub[i, j],
                "shap_value": sv[i, j],
            })

regime_imp_df = pd.DataFrame(regime_rows)
regime_imp_df.to_csv(OUT_DIR / "shap_importancia_por_regime.csv", index=False)
pd.DataFrame(regime_shap_long).to_csv(OUT_DIR / "shap_valores_por_regime.csv", index=False)

if not regime_imp_df.empty:
    plot_regime_importance(regime_imp_df)

    # Diferenca alta - baixa
    piv = regime_imp_df.pivot(index="feature", columns="regime", values="mean_abs_shap").fillna(0)
    if "Alta vazao" in piv.columns and "Baixa vazao" in piv.columns:
        delta = (piv["Alta vazao"] - piv["Baixa vazao"]).sort_values()
        delta_df = pd.DataFrame({
            "feature": delta.index,
            "delta_SHAP_alta_menos_baixa": delta.values,
        })
        delta_df.to_csv(OUT_DIR / "delta_shap_alta_vs_baixa.csv", index=False)

        plt.figure(figsize=(9, max(4, 0.35 * min(15, len(delta_df)))))
        dplot = delta_df.reindex(delta_df["delta_SHAP_alta_menos_baixa"].abs().sort_values(ascending=False).index).head(15)
        dplot = dplot.sort_values("delta_SHAP_alta_menos_baixa")
        plt.barh(dplot["feature"], dplot["delta_SHAP_alta_menos_baixa"])
        plt.axvline(0, linewidth=1)
        plt.xlabel("Delta mean |SHAP| = Alta vazao - Baixa vazao")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "delta_shap_alta_vs_baixa.png", dpi=200)
        plt.close()

# %% ============================================================
# 12. SHAP DEPENDENCE PARA TOP FEATURES
# ===============================================================

try:
    X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)
    for feature in top_shap_vars[:4]:
        shap.dependence_plot(
            feature,
            shap_vals,
            X_shap_df,
            interaction_index=None,
            show=False,
        )
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace("\\", "_")
        plt.savefig(OUT_DIR / f"shap_dependence_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close()
except Exception as e:
    print("Aviso: dependence plots nao gerados:", e)

# %% ============================================================
# 13. EXPLICACOES LOCAIS: EXEMPLO BAIXA / NORMAL / ALTA
# ===============================================================

local_rows = []
for regime_name in ["Baixa vazao", "Normal", "Alta vazao"]:
    idx_candidates = np.where(regime_test == regime_name)[0]
    if len(idx_candidates) == 0:
        continue

    chosen = idx_candidates[:LOCAL_EXAMPLES_PER_REGIME]
    for idx_local in chosen:
        x_one = X_test_all.iloc[[idx_local]].values

        def predict_fn_local(X_np):
            return deterministic_predict(model_all, scaler_all, np.asarray(X_np)).reshape(-1)

        bg_idx = rng.choice(len(X_train_all), size=min(SHAP_BACKGROUND_N, len(X_train_all)), replace=False)
        expl = shap.KernelExplainer(predict_fn_local, X_train_all.values[bg_idx])
        sv = np.asarray(expl.shap_values(x_one, nsamples="auto"))
        if sv.ndim == 3:
            sv = sv[..., 0]
        sv = sv.reshape(-1)

        pred_one = deterministic_predict(model_all, scaler_all, x_one)[0]

        row = {
            "regime": regime_name,
            "data": dates_test[idx_local],
            "observado": y_test[idx_local],
            "previsto": pred_one,
        }
        for f, val in zip(feature_cols, sv):
            row[f"SHAP_{f}"] = val
        local_rows.append(row)

pd.DataFrame(local_rows).to_csv(OUT_DIR / "shap_explicacoes_locais.csv", index=False)

# %% ============================================================
# 14. INCERTEZA EPISTEMICA - MLP-ALL
# ===============================================================

mc_mean, mc_std, mc_lower, mc_upper = mc_dropout_predict(
    model_all,
    scaler_all,
    X_test_all.values,
    n_samples=MC_SAMPLES,
)

coverage = (y_test >= mc_lower) & (y_test <= mc_upper)
abs_error = np.abs(y_test - mc_mean)

uncertainty_df = pd.DataFrame({
    "data": dates_test,
    "observado": y_test,
    "previsao_media": mc_mean,
    "incerteza_epistemica_std": mc_std,
    "limite_inferior_epistemico": mc_lower,
    "limite_superior_epistemico": mc_upper,
    "largura_intervalo_epistemico": mc_upper - mc_lower,
    "erro_absoluto": abs_error,
    "coberto_intervalo": coverage.astype(int),
    "regime": regime_test,
})
uncertainty_df.to_csv(OUT_DIR / "incerteza_epistemica_teste.csv", index=False)

# Grafico geral de incerteza
plt.figure(figsize=(14, 5))
plt.plot(dates_test, y_test, label="Observado", linewidth=1.2)
plt.plot(dates_test, mc_mean, label="Previsao media", linewidth=1.2)
plt.fill_between(dates_test, mc_lower, mc_upper, alpha=0.25, label="Intervalo epistemico 95%")
plt.xlabel("Data")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "previsao_com_incerteza_epistemica.png", dpi=200)
plt.close()

# Erro vs incerteza
corr_err_unc = np.corrcoef(abs_error, mc_std)[0, 1] if len(abs_error) > 1 else np.nan
plt.figure(figsize=(6, 5))
plt.scatter(mc_std, abs_error, alpha=0.4)
plt.xlabel("Incerteza epistemica (std)")
plt.ylabel("Erro absoluto")
plt.tight_layout()
plt.savefig(OUT_DIR / "erro_vs_incerteza.png", dpi=200)
plt.close()

# %% ============================================================
# 15. INCERTEZA POR REGIME HIDROLOGICO
# ===============================================================

regime_unc_rows = []
for regime_name in ["Baixa vazao", "Normal", "Alta vazao"]:
    sub = uncertainty_df[uncertainty_df["regime"] == regime_name]
    if len(sub) == 0:
        continue

    m = calc_metrics(sub["observado"].values, sub["previsao_media"].values)
    regime_unc_rows.append({
        "regime": regime_name,
        "N": len(sub),
        **m,
        "Incerteza_epistemica_media": sub["incerteza_epistemica_std"].mean(),
        "Largura_media_intervalo": sub["largura_intervalo_epistemico"].mean(),
        "Cobertura_intervalo": sub["coberto_intervalo"].mean(),
        "Correlacao_erro_incerteza": (
            np.corrcoef(sub["erro_absoluto"], sub["incerteza_epistemica_std"])[0, 1]
            if len(sub) > 1 else np.nan
        ),
    })

regime_unc_df = pd.DataFrame(regime_unc_rows)
regime_unc_df.to_csv(OUT_DIR / "incerteza_por_regime.csv", index=False)

if not regime_unc_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(regime_unc_df["regime"], regime_unc_df["Incerteza_epistemica_media"])
    plt.ylabel("Incerteza epistemica media (std)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "incerteza_media_por_regime.png", dpi=200)
    plt.close()

# %% ============================================================
# 16. COMPARACAO SHAP X INCERTEZA POR REGIME
# ===============================================================

if not regime_imp_df.empty and not regime_unc_df.empty:
    top_regime = (
        regime_imp_df.sort_values(["regime", "mean_abs_shap"], ascending=[True, False])
        .groupby("regime")
        .head(5)
    )
    top_regime.to_csv(OUT_DIR / "top5_shap_por_regime.csv", index=False)

# %% ============================================================
# 17. PREVISOES DE TODOS OS MODELOS NO TESTE
# ===============================================================

pred_table = pd.DataFrame({
    "data": dates_test,
    "observado": y_test,
    "regime": regime_test,
})

pred_table["MLP_All"] = deterministic_predict(model_all, scaler_all, X_test_all.values)

for label, cols in selection_sets.items():
    pred_table[label.replace("-", "_")] = deterministic_predict(
        models[label], scalers[label], X_test_all[cols].values
    )

pred_table.to_csv(OUT_DIR / "previsoes_todos_modelos.csv", index=False)

# %% ============================================================
# 18. SALVAR MODELOS
# ===============================================================

torch.save(model_all.state_dict(), OUT_DIR / "modelo_mlp_all.pt")
for label, cols in selection_sets.items():
    torch.save(models[label].state_dict(), OUT_DIR / f"modelo_{label.lower().replace('-', '_')}.pt")

# %% ============================================================
# 19. RESUMO FINAL
# ===============================================================

summary = {
    "T": T,
    "TOP_N": TOP_N,
    "Q_low_train": q_low,
    "Q_high_train": q_high,
    "MC_SAMPLES": MC_SAMPLES,
    "Cobertura_global_intervalo": coverage.mean(),
    "Incerteza_epistemica_media_global": mc_std.mean(),
    "Largura_media_intervalo_global": np.mean(mc_upper - mc_lower),
    "Correlacao_erro_incerteza_global": corr_err_unc,
}

pd.DataFrame([summary]).to_csv(OUT_DIR / "resumo_execucao.csv", index=False)

print("\n=== RESUMO ===")
print(metrics_df.round(4))
print("\nTop Corr:", top_corr_vars)
print("Top MI:", top_mi_vars)
print("Top SHAP:", top_shap_vars)
print(f"\nQ{int(LOW_Q*100)} treino = {q_low:.4f}")
print(f"Q{int(HIGH_Q*100)} treino = {q_high:.4f}")
print(f"Cobertura intervalo epistemico = {coverage.mean():.3f}")
print(f"Correlacao erro x incerteza = {corr_err_unc:.3f}")
print(f"\nArquivos salvos em: {OUT_DIR}")
