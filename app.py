import json
from flask import Flask, Response, jsonify
from model import download_data, format_data, train_model, predict_price
from model import forecast_price
import os

app = Flask(__name__)

# Load config.json
config_path = os.getenv('CONFIG_PATH', 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Tokens yang tersedia dari config atau bisa langsung ditetapkan
TOKENS = ["ETH", "BTC", "BNB", "SOL", "ARB"]

def update_data():
    """Download price data, format data, and train the model."""
    try:
        for token in TOKENS:
            download_data(token)  # Mengunduh data terbaru dari Binance
            format_data(token)    # Memformat data yang didownload
            train_model(token)    # Melatih model TinyTimeMixer untuk token tersebut
        return True
    except Exception as e:
        print(f"Error during update: {e}")
        return False

def get_token_inference(token):
    """Return the forecasted price for the given token."""
    if token not in forecast_price:
        predict_price(token)  # Prediksi jika belum ada di forecast_price
    return forecast_price.get(token, 0)

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference (price prediction) for the given token."""
    if not token or token not in TOKENS:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_token_inference(token)
        return jsonify({"token": token, "predicted_price": inference})
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Download new data, train models, and return status."""
    if update_data():
        return Response(json.dumps({"status": "Data updated successfully"}), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"status": "Failed to update data"}), status=500, mimetype='application/json')

if __name__ == "__main__":
    # Update data and models upon starting the app
    update_data()
    app.run(host="0.0.0.0", port=8011)