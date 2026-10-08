# %%
# ================================================================
# Seção 1: Importação de bibliotecas e definição de parâmetros
# ================================================================
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Parâmetros definidos pelo usuário
H = 3         # Janela de histórico: utiliza variáveis dos instantes t, t-1, ..., t-H+1
T = 1         # Horizonte de previsão: prever Q_Afluente no tempo t+T
top_n = 6     # Número de variáveis mais correlacionadas a serem utilizadas

# %%
# ================================================================
# Seção 2: Carregamento do dataset e criação da variável alvo
# ================================================================
# Carrega os dados e define a variável target como Q_Afluente deslocada em T passos no futuro
df = pd.read_csv("dataset_filled.csv").drop(columns=["data"])
df["target"] = df["Q_Afluente"].shift(-T)
df = df.dropna()  # Remove linhas com valores ausentes (causados pelo shift)

# %%
# ================================================================
# Seção 3: Seleção das top-N variáveis com maior correlação com o alvo
# ================================================================
# Calcula a correlação de Pearson entre as variáveis exógenas e o alvo (target)
exog_inputs = df.drop(columns=["Q_Afluente", "target"])
corr_matrix = pd.concat([exog_inputs, df["target"]], axis=1).corr()
target_corr = corr_matrix["target"].drop("target").sort_values(ascending=False)
top_features = target_corr.head(top_n).index.tolist()

# %%
# ================================================================
# Seção 4: Criação de variáveis defasadas (lags)
# ================================================================
# Cria variáveis defasadas a partir das top_features selecionadas (janela de histórico H)

lagged_features = []
for h in range(H):
    lag = exog_inputs[top_features].shift(h)
    lag.columns = [f"{col}_t-{h}" for col in lag.columns]
    lagged_features.append(lag)

# Concatena todas as defasagens e adiciona a variável alvo
df_lagged = pd.concat(lagged_features, axis=1)
df_lagged["target"] = df["target"]
df_lagged.dropna(inplace=True)

# %%
# ================================================================
# Seção 5: Separação em treino/validação e normalização dos dados
# ================================================================
# Separa os dados de entrada e saída
X = df_lagged.drop(columns=["target"]).values
y = df_lagged["target"].values

# Divisão temporal (sem embaralhar)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Padronização dos dados (z-score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Conversão para tensores PyTorch
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# %%
# ================================================================
# Seção 6: Definição da arquitetura do MLP
# ================================================================
# Define a estrutura da rede neural com 2 camadas ocultas e ReLU

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Instancia o modelo
model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# %%
# ================================================================
# Seção 7: Treinamento do modelo
# ================================================================
epochs = 1500
batch_size = 365

for epoch in range(epochs):
    # Embaralha os dados de treino
    indices = torch.randperm(X_train.size(0))
    X_train = X_train[indices]
    y_train = y_train[indices]

    model.train()
    for i in range(0, X_train.size(0), batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # Avaliação no conjunto de validação
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss.item():.4f}")

# %%
# ================================================================
# Seção 8: Avaliação e gráfico para o conjunto de validação
# ================================================================
val_pred_np = val_pred.numpy().flatten()
y_val_np = y_val.numpy().flatten()

mae = mean_absolute_error(y_val_np, val_pred_np)
nse = 1 - np.sum((y_val_np - val_pred_np) ** 2) / np.sum((y_val_np - np.mean(y_val_np)) ** 2)

print(f"\n📊 Métricas de Validação:")
print(f"MAE: {mae:.4f}")
print(f"NSE: {nse:.4f}")

# Gráfico de comparação entre observado e previsto
plt.figure(figsize=(12, 5))
plt.plot(y_val_np, label="Observado", linewidth=1.5)
plt.plot(val_pred_np, label="Previsto", linewidth=1.5)
plt.title(f"Previsão Q_Afluente (t+{T}) - Validação")
plt.xlabel("Índice Temporal")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# ================================================================
# Seção 9: Previsão e gráfico para toda a série
# ================================================================
X_full = df_lagged.drop(columns=["target"]).values
X_full_scaled = scaler.transform(X_full)
X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_full_pred = model(X_full_tensor).numpy().flatten()

y_full_true = df_lagged["target"].values

plt.figure(figsize=(14, 5))
plt.plot(y_full_true, label="Observado", linewidth=1.5)
plt.plot(y_full_pred, label="Previsto", linewidth=1.5)
plt.title(f"Previsão Q_Afluente (t+{T}) - Série Completa")
plt.xlabel("Índice Temporal")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
