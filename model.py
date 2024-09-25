import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import random
import requests
import retrying
from modeling_tinytimemixer import TinyTimeMixerModel  # Import TinyTimeMixerModel
from configuration_tinytimemixer import TinyTimeMixerConfig  # Import TinyTimeMixerConfig
from sklearn.preprocessing import MinMaxScaler
import pickle

# Path untuk menyimpan data dari Binance dan model
from config import data_base_path
binance_data_path = os.path.join(data_base_path, "binance/futures-klines")

MAX_DATA_SIZE = 100  # Batas maksimum data yang akan disimpan
INITIAL_FETCH_SIZE = 100  # Jumlah data yang diambil pertama kali

# Global variable untuk menyimpan hasil prediksi
forecast_price = {}

# Tentukan device untuk GPU atau CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load atau inisialisasi TinyTimeMixerModel
def load_model(model_path=None):
    # Membuat konfigurasi TinyTimeMixer
    config = TinyTimeMixerConfig(
        context_length=64,        # Panjang konteks input (history length)
        patch_length=8,           # Panjang patch untuk input sequence
        num_input_channels=1,     # Jumlah variabel input (univariat)
        prediction_length=1,      # Panjang prediksi (berapa harga ke depan)
        d_model=16,               # Ukuran fitur tersembunyi (hidden size)
        num_layers=3,             # Jumlah lapisan dalam model
        dropout=0.2,              # Dropout untuk regularisasi
        mode="common_channel"     # Mode pemrosesan channel
    )
    
    # Inisialisasi model TinyTimeMixerModel dengan konfigurasi
    model = TinyTimeMixerModel(config)
    
    # Jika model yang dilatih sudah ada, load dari model_path
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print("Initialized a new TinyTimeMixerModel")
    
    # Pindahkan model ke device (GPU atau CPU)
    model.to(device)
    
    return model

# Fungsi untuk fetch data dari Binance
@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=100, start_time=None, end_time=None):
    try:
        base_url = "https://fapi.binance.com"
        endpoint = "/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch prices for {symbol}: {str(e)}")
        raise e

# Download dan update data
def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    # Check jika file data sudah ada
    if os.path.exists(file_path):
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE * 5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df])
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

# Format data untuk pelatihan model
def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return None

    df = pd.read_csv(file_path)

    columns_to_use = ["start_time", "close"]
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df["date"] = pd.to_datetime(df["start_time"], unit='ms')
        df.set_index("date", inplace=True)
        return df["close"].values.reshape(-1, 1)  # Ambil harga close
    else:
        print(f"Required columns missing in {file_path}")
        return None

# Fungsi untuk melatih model menggunakan TinyTimeMixerModel
def train_model(token):
    # Load data
    price_data = format_data(token)
    
    if price_data is None:
        print(f"No data available for {token}. Skipping training.")
        return
    
    # Preprocessing: Scaling data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(price_data)
    
    # Simpan scaler untuk prediksi di masa depan
    scaler_path = os.path.join(data_base_path, f"{token.lower()}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Pastikan ada cukup data untuk melakukan pelatihan
    if len(scaled_data) < 65:  # Karena kita membutuhkan setidaknya 64 data untuk X dan 64 data untuk y
        print("Not enough data for training, need at least 65 rows.")
        return

    # Inisialisasi model
    model = load_model()

    # Persiapkan data untuk pelatihan, ambil 64 elemen terakhir untuk X dan 64 elemen terakhir untuk y
    X = torch.tensor(scaled_data[-64:], dtype=torch.float32).unsqueeze(0).to(device)  # Mengambil 64 elemen terakhir
    y = torch.tensor(scaled_data[-64:], dtype=torch.float32).unsqueeze(0).to(device)    # Mengambil 64 elemen terakhir

    # Log untuk memastikan ukuran data
    print(f"Shape of input (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    # Optimizer dan loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Pelatihan
    model.train()
    epochs = 50  # Jumlah iterasi pelatihan
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs.predictions, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    # Simpan model terlatih
    model_path = os.path.join(data_base_path, f"{token.lower()}_tinytimemixer.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def predict_price(token):
    # Load data
    price_data = format_data(token)

    if price_data is None:
        print(f"No data available for {token}. Cannot predict.")
        return

    # Load scaler dan model
    scaler_path = os.path.join(data_base_path, f"{token.lower()}_scaler.pkl")
    model_path = os.path.join(data_base_path, f"{token.lower()}_tinytimemixer.pth")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = load_model(model_path)
    model.eval()

    # Preprocessing: Scale the input data
    scaled_data = scaler.transform(price_data[-1].reshape(1, -1))

    # Convert to tensor for prediction, tambahkan dimensi batch
    X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict the next price
    with torch.no_grad():
        predicted_scaled = model(X)
    
    # Inverse scale the predicted price
    predicted_price = scaler.inverse_transform(predicted_scaled.cpu().numpy())[0][0]

    # Forecasted price with small fluctuation
    fluctuation = 0.001 * predicted_price
    forecast_price[token] = random.uniform(predicted_price - fluctuation, predicted_price + fluctuation)

    print(f"Forecasted price for {token}: {forecast_price[token]}")

def update_data():
    tokens = ["ETH", "BTC", "BNB", "SOL", "ARB"]
    for token in tokens:
        download_data(token)
        train_model(token)
        predict_price(token)

if __name__ == "__main__":
    update_data()
