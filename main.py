import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import math

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading & Preprocessing ---
try:
    df_energy = pd.read_csv('energy_dataset.csv')
    df_weather = pd.read_csv('weather_features.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Using dummy data.")
    dates = pd.date_range(start='2022-01-01', periods=1000, freq='H')
    df_energy = pd.DataFrame({'time': dates, 'total load actual': np.random.rand(1000)*100 + 200, 'price actual': np.random.rand(1000)*50})
    df_weather = pd.DataFrame({'dt_iso': dates, 'city_name': 'Valencia', 'temp': np.random.rand(1000)*30, 'humidity': np.random.rand(1000)*100, 'wind_speed': np.random.rand(1000)*10})

df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)

df_weather = df_weather[df_weather['city_name'] == 'Valencia']
df_weather = df_weather.drop_duplicates(subset=['dt_iso'])

df = pd.merge(df_energy, df_weather, left_on='time', right_on='dt_iso', how='inner')

features = ['total load actual', 'price actual', 'temp', 'humidity', 'wind_speed']
df = df[features].copy()

df = df.interpolate(method='linear').dropna()

# --- 2. Data Normalization ---
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# --- 3. Sliding Window ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 24
X, y = create_sequences(df_scaled, SEQ_LENGTH)

train_size = int(len(X) * 0.8)
X_train_np, X_test_np = X[:train_size], X[train_size:]
y_train_np, y_test_np = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

BATCH_SIZE = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[2]


# --- Model Definitions ---

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# --- Optimized Transformer ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Optimization: Explicitly set dim_feedforward to 4*d_model (256) instead of default 2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 256
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        output = self.fc(x)
        return output


# --- Training Helper with Scheduler ---
def train_model(model, train_loader, epochs=50, lr=0.001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Add Scheduler: Reduce LR if loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # Step the scheduler
        scheduler.step(avg_loss)

    return loss_history


# --- Evaluation Helper ---
def evaluate_model(model, test_loader, scaler, original_features_count):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            predictions.append(preds.cpu().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    dummy_pred = np.zeros((len(predictions), original_features_count))
    dummy_actual = np.zeros((len(actuals), original_features_count))

    dummy_pred[:, 0] = predictions[:, 0]
    dummy_actual[:, 0] = actuals[:, 0]

    real_pred = scaler.inverse_transform(dummy_pred)[:, 0]
    real_actual = scaler.inverse_transform(dummy_actual)[:, 0]

    rmse = np.sqrt(mean_squared_error(real_actual, real_pred))
    mae = mean_absolute_error(real_actual, real_pred)
    r2 = r2_score(real_actual, real_pred)

    return rmse, mae, r2


# --- Main Execution ---
EPOCHS = 100
results = {}
history_dict = {}

models_dict = {
    'RNN': RNNModel(input_dim),
    'LSTM': LSTMModel(input_dim),
    # Slightly deeper transformer
    'Transformer': TransformerModel(input_dim, d_model=64, nhead=4, num_layers=3)
}

print("Starting training with optimized hyperparameters...")

for name, model in models_dict.items():
    print(f"\nTraining {name} on {device}...")
    start_time = time.time()

    losses = train_model(model, train_loader, epochs=EPOCHS)
    rmse, mae, r2 = evaluate_model(model, test_loader, scaler, df_scaled.shape[1])

    history_dict[name] = losses
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}

    print(f"{name} Done. RMSE: {rmse:.2f}, Time: {time.time() - start_time:.1f}s")



# --- Visualization & Results ---
plt.figure(figsize=(10, 6))
plt.plot(history_dict['RNN'], label='RNN Loss', linestyle='--', alpha=0.7)
plt.plot(history_dict['LSTM'], label='LSTM Loss', linestyle='-.', alpha=0.7)
plt.plot(history_dict['Transformer'], label='Transformer Loss', linewidth=2, color='red')
plt.title('Model Training Loss Comparison (Optimized)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='R2 Score', ascending=False)

print("\n=== Final Model Performance ===")
print(results_df)
print("\nMarkdown Table:")
print(results_df.to_markdown())