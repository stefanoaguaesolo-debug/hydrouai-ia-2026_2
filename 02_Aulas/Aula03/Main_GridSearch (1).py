# %% ==========================================================
# Seção 1: Importações e Parâmetros Globais
# =============================================================
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import tqdm

T = 1  # Horizonte de previsão
epochs = 2000
batch_size = 365
layer_options = [2, 3]
neuron_options = [32, 64,128]

# %% ==========================================================
# Seção 2: Carregamento e Pré-processamento dos Dados
# =============================================================
df = pd.read_csv("dataset_filled.csv").drop(columns=["data"])
df["target"] = df["Q_Afluente"].shift(-T)
df.dropna(inplace=True)

X = df.drop(columns=["Q_Afluente", "target"]).values
y = df["target"].values

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# %% ==========================================================
# Seção 3: Busca em Grade (Grid Search) de Arquiteturas
# =============================================================
results = []

for num_layers, num_neurons in tqdm(product(layer_options, neuron_options), total=len(layer_options)*len(neuron_options)):
    # Criação dinâmica da MLP
    layers = [nn.Linear(X.shape[1], num_neurons), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
    layers.append(nn.Linear(num_neurons, 1))

    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Treinamento
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[indices]
        y_train_tensor = y_train_tensor[indices]

        for i in range(0, X_train_tensor.size(0), batch_size):
            xb = X_train_tensor[i:i+batch_size]
            yb = y_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Avaliação
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val_tensor).numpy().flatten()
        y_val_np = y_val_tensor.numpy().flatten()
        mae = mean_absolute_error(y_val_np, preds_val)
        nse = 1 - np.sum((y_val_np - preds_val)**2) / np.sum((y_val_np - np.mean(y_val_np))**2)

    results.append({
        'layers': num_layers,
        'neurons': num_neurons,
        'MAE': mae,
        'NSE': nse,
        'model_state': model.state_dict()
    })

# %% ==========================================================
# Seção 4: Visualização dos Resultados
# =============================================================
results_df = pd.DataFrame(results)
pivot_nse = results_df.pivot(index="layers", columns="neurons", values="NSE")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_nse, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("NSE para diferentes arquiteturas MLP")
plt.ylabel("Número de camadas ocultas")
plt.xlabel("Neurônios por camada")
plt.tight_layout()
plt.show()

# %% ==========================================================
# Seção 5: Previsão na Série Completa com Melhor Arquitetura
# =============================================================
best_row = results_df.sort_values("NSE", ascending=False).iloc[0]
best_layers = best_row["layers"]
best_neurons = best_row["neurons"]

# Reconstrói modelo com melhor arquitetura
layers = [nn.Linear(X.shape[1], best_neurons), nn.ReLU()]
for _ in range(best_layers - 1):
    layers += [nn.Linear(best_neurons, best_neurons), nn.ReLU()]
layers.append(nn.Linear(best_neurons, 1))

best_model = nn.Sequential(*layers)
best_model.load_state_dict(best_row["model_state"])
best_model.eval()

# Previsão para o dataset completo
X_full_scaled = scaler.transform(X)
X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)

with torch.no_grad():
    y_full_pred = best_model(X_full_tensor).numpy().flatten()

# Plot da série observada vs prevista
plt.figure(figsize=(14, 5))
plt.plot(y, label="Observado", linewidth=1.5)
plt.plot(y_full_pred, label="Previsto", linewidth=1.5)
plt.title(f"Previsão da Série Completa com Melhor MLP (t+{T})")
plt.xlabel("Índice Temporal")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
