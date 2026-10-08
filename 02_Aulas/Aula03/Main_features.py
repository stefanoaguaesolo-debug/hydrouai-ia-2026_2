# %%
# ============================================================
# Seção 1: Importações e Configurações Iniciais
# ============================================================
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
T = 1          # Horizonte de previsão
top_n = 5      # Número de variáveis mais importantes a serem selecionadas
epochs = 1500
batch_size = 365*2
lr = 0.0001

# %%
# ============================================================
# Seção 2: Carregamento dos Dados e Preparação
# ============================================================
df = pd.read_csv("dataset_filled.csv").drop(columns=["data"])
df["target"] = df["Q_Afluente"].shift(-T)
df.dropna(inplace=True)

X_raw = df.drop(columns=["Q_Afluente", "target"])
y = df["target"].values

# Split temporal
split_idx = int(len(X_raw) * 0.8)
X_train_all, X_val_all = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# %%
# ============================================================
# Seção 3: Seleção de Variáveis - Correlação e Informação Mútua
# ============================================================
# Correlação
corr_matrix = pd.concat([X_raw, df["target"]], axis=1).corr()
target_corr = corr_matrix["target"].drop("target").sort_values(ascending=False)
top_corr_vars = target_corr.head(top_n).index.tolist()

# Informação Mútua
mi_scores = mutual_info_regression(X_raw, df["target"])
mi_series = pd.Series(mi_scores, index=X_raw.columns).sort_values(ascending=False)
top_mi_vars = mi_series.head(top_n).index.tolist()

print("🔍 Top correlação:", top_corr_vars)
print("🔍 Top informação mútua:", top_mi_vars)

# %%
# ============================================================
# Seção 4: Função Auxiliar - Treinar e Avaliar MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_evaluate_mlp(X_train, X_val, y_train, y_val, label):
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Tensores
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = MLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    val_losses = []
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[idx]
        y_train_tensor = y_train_tensor[idx]

        for i in range(0, X_train_tensor.size(0), batch_size):
            xb = X_train_tensor[i:i+batch_size]
            yb = y_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
            val_losses.append(val_loss)

    # Avaliação final
    val_pred_np = val_pred.numpy().flatten()
    y_val_np = y_val_tensor.numpy().flatten()

    mae = mean_absolute_error(y_val_np, val_pred_np)
    nse = 1 - np.sum((y_val_np - val_pred_np)**2) / np.sum((y_val_np - np.mean(y_val_np))**2)

    print(f"\n📊 [{label}] MAE: {mae:.4f} | NSE: {nse:.4f}")
    return val_losses, mae, nse

# %%
# ============================================================
# Seção 5: Treinamento dos Modelos
# ============================================================
loss_all, mae_all, nse_all = train_evaluate_mlp(X_train_all, X_val_all, y_train, y_val, "Todos os Inputs")
loss_corr, mae_corr, nse_corr = train_evaluate_mlp(X_train_all[top_corr_vars], X_val_all[top_corr_vars], y_train, y_val, "Top Correlação")
loss_mi, mae_mi, nse_mi = train_evaluate_mlp(X_train_all[top_mi_vars], X_val_all[top_mi_vars], y_train, y_val, "Top Info Mútua")

# %%
# ============================================================
# Seção 6: Comparação das Curvas de Validação
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(loss_all, label="Todos os Inputs")
plt.plot(loss_corr, label="Top Correlação")
plt.plot(loss_mi, label="Top Info Mútua")
plt.title("Curva de Validação (Loss por Época)")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ============================================================
# Seção 7: Tabela Comparativa de Desempenho
# ============================================================
# Imprime uma tabela resumida das métricas de cada abordagem

print("\n📋 Comparação Final de Desempenho")
print(pd.DataFrame({
    "MAE": [mae_all, mae_corr, mae_mi],
    "NSE": [nse_all, nse_corr, nse_mi]
}, index=["Todos os Inputs", "Top Correlação", "Top Info Mútua"]).round(4))

