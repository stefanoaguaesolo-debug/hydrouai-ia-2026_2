# %% ============================================================================
# LSTM + INCERTEZA EPISTEMICA vs MLP - PREVISAO DE Q_AFLUENTE
# Arquivo independente, adaptado para series_preenchidas.csv
# ============================================================================

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %% ============================================================================
# 1. PARAMETROS DO USUARIO
# ============================================================================

SEED = 42
H = 7                          # janela historica: t, t-1, ..., t-H+1
T = 1                          # horizonte: prever Q_Afluente em t+T
TOP_N = 6                      # numero de variaveis exogenas selecionadas
INCLUDE_Q_AFLUENTE = True      # inclui Q_Afluente na sequencia de entrada

TRAIN_FRAC = 0.70              # treino
TUNE_FRAC = 0.15               # ajuste / early stopping
# 15% finais = teste independente

MAX_EPOCHS = 300
BATCH_SIZE = 365
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 120
MIN_DELTA = 1e-6

# LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.20

# MLP de comparacao
MLP_HIDDEN_UNITS = [128, 64]
MLP_DROPOUT = 0.20

# Monte Carlo Dropout - incerteza epistemica
MC_SAMPLES = 300
INTERVAL = 0.95                # intervalo epistemico de 95%

TARGET_COL = "Q_Afluente"
DATE_COL = "data"

# Caminhos relativos: CSV na mesma pasta do script.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "series_preenchidas.csv"
OUTPUT_DIR = BASE_DIR / "resultados_lstm_incerteza"
OUTPUT_DIR.mkdir(exist_ok=True)


# %% ============================================================================
# 2. REPRODUTIBILIDADE E DISPOSITIVO
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")


# %% ============================================================================
# 3. FUNCOES AUXILIARES
# ============================================================================

def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / den


def metrics(obs, sim):
    return {
        "MAE": mean_absolute_error(obs, sim),
        "RMSE": np.sqrt(mean_squared_error(obs, sim)),
        "NSE": nse(obs, sim),
        "R2": r2_score(obs, sim),
    }


def uncertainty_metrics(obs, mean_pred, lower, upper, epistemic_std):
    obs = np.asarray(obs)
    mean_pred = np.asarray(mean_pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    epistemic_std = np.asarray(epistemic_std)

    coverage = np.mean((obs >= lower) & (obs <= upper))
    mean_width = np.mean(upper - lower)
    abs_error = np.abs(obs - mean_pred)

    if np.std(abs_error) > 0 and np.std(epistemic_std) > 0:
        error_unc_corr = np.corrcoef(abs_error, epistemic_std)[0, 1]
    else:
        error_unc_corr = np.nan

    return {
        "Cobertura_intervalo": coverage,
        "Largura_media_intervalo": mean_width,
        "Incerteza_epistemica_media": np.mean(epistemic_std),
        "Correlacao_erro_incerteza": error_unc_corr,
    }


def enable_mc_dropout(model):
    """Mantem o modelo em eval(), mas reativa somente as camadas Dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, X_tensor, n_samples=300, interval=0.95):
    """Executa multiplas previsoes com Dropout ativo e retorna a distribuicao epistemica."""
    enable_mc_dropout(model)
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).detach().cpu().numpy().ravel()
            samples.append(pred)

    samples = np.asarray(samples)
    alpha = 1.0 - interval

    mean_pred = samples.mean(axis=0)
    std_pred = samples.std(axis=0, ddof=1)
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)

    model.eval()
    return mean_pred, std_pred, lower, upper


# %% ============================================================================
# 4. MODELOS
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        # Dropout interno da LSTM so atua entre camadas quando num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Dropout explicito na representacao final: essencial para MC Dropout.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim

        for units in hidden_units:
            layers.extend([
                nn.Linear(in_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = units

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# %% ============================================================================
# 5. TREINAMENTO GENERICO COM EARLY STOPPING
# ============================================================================

def train_model(model, X_train, y_train, X_tune, y_tune, model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_tune_loss = np.inf
    wait = 0
    train_history = []
    tune_history = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.shape[0], device=DEVICE)
        batch_losses = []

        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            tune_loss = criterion(model(X_tune), y_tune).item()

        train_history.append(train_loss)
        tune_history.append(tune_loss)

        if tune_loss < best_tune_loss - MIN_DELTA:
            best_tune_loss = tune_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(
                f"{model_name} | Epoca {epoch + 1:4d} | "
                f"MSE treino={train_loss:.5f} | MSE ajuste={tune_loss:.5f}"
            )

        if wait >= PATIENCE:
            print(f"{model_name} | Early stopping na epoca {epoch + 1}.")
            break

    model.load_state_dict(best_state)
    return model, train_history, tune_history


# %% ============================================================================
# 6. LEITURA DOS DADOS E CRIACAO DO ALVO
# ============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo nao encontrado: {DATA_FILE}\n"
        "Coloque 'series_preenchidas.csv' na mesma pasta do script."
    )

raw = pd.read_csv(DATA_FILE)

required = {DATE_COL, TARGET_COL}
missing = required.difference(raw.columns)
if missing:
    raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")

raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
raw = raw.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
exog_cols = [c for c in numeric_cols if c != TARGET_COL]

if len(exog_cols) == 0:
    raise ValueError("Nenhuma variavel exogena numerica foi encontrada.")

base = raw[[DATE_COL, TARGET_COL] + exog_cols].copy()
base["target"] = base[TARGET_COL].shift(-T)
base["data_target"] = base[DATE_COL].shift(-T)
base = base.dropna(subset=["target", "data_target"]).reset_index(drop=True)


# %% ============================================================================
# 7. SELECAO DAS VARIAVEIS - APENAS NO TREINO
# ============================================================================

# O corte e calculado antes da montagem das sequencias apenas para impedir que
# informacao futura participe da selecao das variaveis.
pre_train_end = int(len(base) * TRAIN_FRAC)
train_for_selection = base.iloc[:pre_train_end].copy()

corr_values = {}
for col in exog_cols:
    pair = train_for_selection[[col, "target"]].dropna()
    if len(pair) > 2:
        corr_values[col] = pair[col].corr(pair["target"])
    else:
        corr_values[col] = np.nan

corr_series = pd.Series(corr_values, name="correlacao").dropna()
corr_table = pd.DataFrame({
    "variavel": corr_series.index,
    "correlacao": corr_series.values,
    "correlacao_abs": np.abs(corr_series.values),
}).sort_values("correlacao_abs", ascending=False)

TOP_N_EFFECTIVE = min(TOP_N, len(corr_table))
top_features = corr_table.head(TOP_N_EFFECTIVE)["variavel"].tolist()
corr_table.to_csv(OUTPUT_DIR / "ranking_correlacao_treino.csv", index=False)

sequence_features = top_features.copy()
if INCLUDE_Q_AFLUENTE:
    sequence_features.append(TARGET_COL)

print("\nVariaveis selecionadas no periodo de treino:")
print(corr_table.head(TOP_N_EFFECTIVE).to_string(index=False))
print(f"\nVariaveis fornecidas aos modelos: {sequence_features}")


# %% ============================================================================
# 8. CONSTRUCAO DAS SEQUENCIAS TEMPORAIS
# ============================================================================

def build_sequences(df, feature_cols, H, target_col="target"):
    X_seq = []
    y_seq = []
    data_origem = []
    data_target = []

    # Para cada linha i, usa [i-H+1, ..., i] para prever target da linha i.
    for i in range(H - 1, len(df)):
        window = df.loc[i - H + 1:i, feature_cols].to_numpy(dtype=np.float32)

        if np.isnan(window).any() or pd.isna(df.loc[i, target_col]):
            continue

        X_seq.append(window)
        y_seq.append(float(df.loc[i, target_col]))
        data_origem.append(df.loc[i, DATE_COL])
        data_target.append(df.loc[i, "data_target"])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        pd.to_datetime(data_origem),
        pd.to_datetime(data_target),
    )


X_seq_raw, y, dates_origin, dates_target = build_sequences(
    base,
    sequence_features,
    H,
)

if len(X_seq_raw) < 30:
    raise ValueError("Poucos registros validos apos a criacao das sequencias.")

n_total = len(y)
idx_train_end = int(n_total * TRAIN_FRAC)
idx_tune_end = int(n_total * (TRAIN_FRAC + TUNE_FRAC))

if not (0 < idx_train_end < idx_tune_end < n_total):
    raise ValueError("Fracoes de treino/ajuste/teste invalidas para o tamanho da serie.")

# Divisao cronologica.
X_train_raw = X_seq_raw[:idx_train_end]
X_tune_raw = X_seq_raw[idx_train_end:idx_tune_end]
X_test_raw = X_seq_raw[idx_tune_end:]

y_train = y[:idx_train_end]
y_tune = y[idx_train_end:idx_tune_end]
y_test = y[idx_tune_end:]


# %% ============================================================================
# 9. NORMALIZACAO SEM VAZAMENTO TEMPORAL
# ============================================================================

# O StandardScaler e ajustado SOMENTE com os dados do treino.
# Para sequencias, juntamos temporariamente os passos de tempo.
n_features = X_seq_raw.shape[2]
scaler = StandardScaler()
scaler.fit(X_train_raw.reshape(-1, n_features))


def scale_sequences(X):
    shape = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_features))
    return X_scaled.reshape(shape).astype(np.float32)


X_train = scale_sequences(X_train_raw)
X_tune = scale_sequences(X_tune_raw)
X_test = scale_sequences(X_test_raw)
X_full = scale_sequences(X_seq_raw)

# LSTM: [amostras, passos_temporais, variaveis]
X_train_lstm_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
X_tune_lstm_t = torch.tensor(X_tune, dtype=torch.float32, device=DEVICE)
X_test_lstm_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
X_full_lstm_t = torch.tensor(X_full, dtype=torch.float32, device=DEVICE)

# MLP: recebe a MESMA janela, apenas achatada.
X_train_mlp = X_train.reshape(len(X_train), -1)
X_tune_mlp = X_tune.reshape(len(X_tune), -1)
X_test_mlp = X_test.reshape(len(X_test), -1)
X_full_mlp = X_full.reshape(len(X_full), -1)

X_train_mlp_t = torch.tensor(X_train_mlp, dtype=torch.float32, device=DEVICE)
X_tune_mlp_t = torch.tensor(X_tune_mlp, dtype=torch.float32, device=DEVICE)
X_test_mlp_t = torch.tensor(X_test_mlp, dtype=torch.float32, device=DEVICE)
X_full_mlp_t = torch.tensor(X_full_mlp, dtype=torch.float32, device=DEVICE)

y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).view(-1, 1)
y_tune_t = torch.tensor(y_tune, dtype=torch.float32, device=DEVICE).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE).view(-1, 1)

print(f"\nH={H} | T={T} | TOP_N={TOP_N_EFFECTIVE}")
print(f"Formato LSTM: {X_train.shape}")
print(f"Formato MLP:  {X_train_mlp.shape}")
print(f"Treino: {len(y_train)} | Ajuste: {len(y_tune)} | Teste: {len(y_test)}")


# %% ============================================================================
# 10. TREINAMENTO DA LSTM
# ============================================================================

set_seed(SEED)
lstm_model = LSTMModel(
    input_size=n_features,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT,
).to(DEVICE)

lstm_model, lstm_train_history, lstm_tune_history = train_model(
    lstm_model,
    X_train_lstm_t,
    y_train_t,
    X_tune_lstm_t,
    y_tune_t,
    "LSTM",
)


# %% ============================================================================
# 11. TREINAMENTO DA MLP DE COMPARACAO
# ============================================================================

set_seed(SEED)
mlp_model = MLPModel(
    input_dim=X_train_mlp.shape[1],
    hidden_units=MLP_HIDDEN_UNITS,
    dropout=MLP_DROPOUT,
).to(DEVICE)

mlp_model, mlp_train_history, mlp_tune_history = train_model(
    mlp_model,
    X_train_mlp_t,
    y_train_t,
    X_tune_mlp_t,
    y_tune_t,
    "MLP",
)


# %% ============================================================================
# 12. INCERTEZA EPISTEMICA NO TESTE - MONTE CARLO DROPOUT
# ============================================================================

print(f"\nExecutando Monte Carlo Dropout com {MC_SAMPLES} amostras por instante...")

lstm_mean, lstm_std, lstm_lower, lstm_upper = mc_dropout_predict(
    lstm_model,
    X_test_lstm_t,
    n_samples=MC_SAMPLES,
    interval=INTERVAL,
)

mlp_mean, mlp_std, mlp_lower, mlp_upper = mc_dropout_predict(
    mlp_model,
    X_test_mlp_t,
    n_samples=MC_SAMPLES,
    interval=INTERVAL,
)


# %% ============================================================================
# 13. METRICAS DE DESEMPENHO E INCERTEZA
# ============================================================================

lstm_metrics = metrics(y_test, lstm_mean)
lstm_unc_metrics = uncertainty_metrics(
    y_test, lstm_mean, lstm_lower, lstm_upper, lstm_std
)

mlp_metrics = metrics(y_test, mlp_mean)
mlp_unc_metrics = uncertainty_metrics(
    y_test, mlp_mean, mlp_lower, mlp_upper, mlp_std
)

comparison = pd.DataFrame([
    {
        "Modelo": "LSTM",
        **lstm_metrics,
        **lstm_unc_metrics,
    },
    {
        "Modelo": "MLP",
        **mlp_metrics,
        **mlp_unc_metrics,
    },
])

comparison.to_csv(OUTPUT_DIR / "comparacao_lstm_mlp.csv", index=False)

print("\nCOMPARACAO NO TESTE INDEPENDENTE")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# %% ============================================================================
# 14. PREVISOES E INCERTEZAS NA SERIE COMPLETA
# ============================================================================

lstm_full_mean, lstm_full_std, lstm_full_lower, lstm_full_upper = mc_dropout_predict(
    lstm_model,
    X_full_lstm_t,
    n_samples=MC_SAMPLES,
    interval=INTERVAL,
)

mlp_full_mean, mlp_full_std, mlp_full_lower, mlp_full_upper = mc_dropout_predict(
    mlp_model,
    X_full_mlp_t,
    n_samples=MC_SAMPLES,
    interval=INTERVAL,
)

sets = np.full(n_total, "teste", dtype=object)
sets[:idx_train_end] = "treino"
sets[idx_train_end:idx_tune_end] = "ajuste"

saida = pd.DataFrame({
    "data_origem": dates_origin,
    "data_target": dates_target,
    "conjunto": sets,
    "Q_Afluente_observado": y,
    "LSTM_previsao_media": lstm_full_mean,
    "LSTM_incerteza_epistemica_std": lstm_full_std,
    "LSTM_limite_inferior": lstm_full_lower,
    "LSTM_limite_superior": lstm_full_upper,
    "LSTM_largura_intervalo": lstm_full_upper - lstm_full_lower,
    "MLP_previsao_media": mlp_full_mean,
    "MLP_incerteza_epistemica_std": mlp_full_std,
    "MLP_limite_inferior": mlp_full_lower,
    "MLP_limite_superior": mlp_full_upper,
    "MLP_largura_intervalo": mlp_full_upper - mlp_full_lower,
})

saida.to_csv(OUTPUT_DIR / "previsoes_incerteza_lstm_vs_mlp.csv", index=False)


# %% ============================================================================
# 15. SALVAMENTO DOS MODELOS
# ============================================================================

torch.save({
    "model_state_dict": lstm_model.state_dict(),
    "model_type": "LSTM",
    "input_size": n_features,
    "hidden_size": LSTM_HIDDEN_SIZE,
    "num_layers": LSTM_NUM_LAYERS,
    "dropout": LSTM_DROPOUT,
    "sequence_features": sequence_features,
    "top_features": top_features,
    "H": H,
    "T": T,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
}, OUTPUT_DIR / "modelo_lstm.pt")


torch.save({
    "model_state_dict": mlp_model.state_dict(),
    "model_type": "MLP",
    "input_dim": X_train_mlp.shape[1],
    "hidden_units": MLP_HIDDEN_UNITS,
    "dropout": MLP_DROPOUT,
    "sequence_features": sequence_features,
    "top_features": top_features,
    "H": H,
    "T": T,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
}, OUTPUT_DIR / "modelo_mlp_comparacao.pt")


# %% ============================================================================
# 16. FIGURAS - CONVERGENCIA
# ============================================================================

plt.figure(figsize=(10, 5))
plt.plot(lstm_train_history, label="LSTM - treino", linewidth=1.2)
plt.plot(lstm_tune_history, label="LSTM - ajuste", linewidth=1.2)
plt.plot(mlp_train_history, label="MLP - treino", linewidth=1.2)
plt.plot(mlp_tune_history, label="MLP - ajuste", linewidth=1.2)
plt.xlabel("Epoca")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convergencia_lstm_vs_mlp.png", dpi=220)
plt.show()


# %% ============================================================================
# 17. FIGURA - LSTM COM INCERTEZA EPISTEMICA NO TESTE
# ============================================================================

test_dates = dates_target[idx_tune_end:]

plt.figure(figsize=(15, 6))
plt.plot(test_dates, y_test, label="Observado", linewidth=1.4)
plt.plot(test_dates, lstm_mean, label="LSTM - media", linewidth=1.2)
plt.fill_between(
    test_dates,
    lstm_lower,
    lstm_upper,
    alpha=0.25,
    label=f"Intervalo epistemico {int(INTERVAL * 100)}%",
)
plt.xlabel("Data")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lstm_incerteza_epistemica_teste.png", dpi=220)
plt.show()


# %% ============================================================================
# 18. FIGURA - MLP COM INCERTEZA EPISTEMICA NO TESTE
# ============================================================================

plt.figure(figsize=(15, 6))
plt.plot(test_dates, y_test, label="Observado", linewidth=1.4)
plt.plot(test_dates, mlp_mean, label="MLP - media", linewidth=1.2)
plt.fill_between(
    test_dates,
    mlp_lower,
    mlp_upper,
    alpha=0.25,
    label=f"Intervalo epistemico {int(INTERVAL * 100)}%",
)
plt.xlabel("Data")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mlp_incerteza_epistemica_teste.png", dpi=220)
plt.show()


# %% ============================================================================
# 19. FIGURA - COMPARACAO DIRETA DAS INCERTEZAS
# ============================================================================

plt.figure(figsize=(15, 5))
plt.plot(test_dates, lstm_std, label="LSTM - desvio padrao epistemico", linewidth=1.2)
plt.plot(test_dates, mlp_std, label="MLP - desvio padrao epistemico", linewidth=1.2)
plt.xlabel("Data")
plt.ylabel("Incerteza epistemica (desvio padrao)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparacao_incerteza_epistemica.png", dpi=220)
plt.show()


# %% ============================================================================
# 20. FIGURA - ERRO ABSOLUTO vs INCERTEZA
# ============================================================================

plt.figure(figsize=(8, 6))
plt.scatter(lstm_std, np.abs(y_test - lstm_mean), alpha=0.45, label="LSTM")
plt.scatter(mlp_std, np.abs(y_test - mlp_mean), alpha=0.45, label="MLP")
plt.xlabel("Incerteza epistemica (desvio padrao)")
plt.ylabel("Erro absoluto")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "erro_vs_incerteza.png", dpi=220)
plt.show()


# %% ============================================================================
# 21. FIGURA - METRICAS COMPARATIVAS
# ============================================================================

metric_plot = comparison.set_index("Modelo")[[
    "MAE",
    "RMSE",
    "Incerteza_epistemica_media",
    "Largura_media_intervalo",
]]

ax = metric_plot.plot(kind="bar", figsize=(10, 6))
ax.set_xlabel("Modelo")
ax.set_ylabel("Valor")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metricas_lstm_vs_mlp.png", dpi=220)
plt.show()


# %% ============================================================================
# 22. RESUMO FINAL
# ============================================================================

print("\nInterpretacao da incerteza:")
print(
    "- O desvio padrao e os limites sao calculados a partir da distribuicao de "
    "previsoes obtida por Monte Carlo Dropout."
)
print(
    "- Essa faixa representa INCERTEZA EPISTEMICA: sensibilidade da previsao aos "
    "parametros/representacao aprendida pelo modelo."
)
print(
    "- Ela NAO representa, isoladamente, toda a incerteza preditiva, pois nao modela "
    "explicitamente a incerteza aleatoria (aleatorica) dos dados."
)
print(
    "- Para comparar LSTM e MLP, observe conjuntamente NSE/RMSE, cobertura, largura "
    "do intervalo e correlacao entre erro absoluto e incerteza."
)

print(f"\nResultados salvos em: {OUTPUT_DIR}")
