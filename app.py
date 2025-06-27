from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carrega modelo treinado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "API de Previsão de Aluguel de Imóveis - São Paulo"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "condominio": float(data["condominio"]),
            "banheiros": int(data["banheiros"]),
            "suites": int(data["suites"]),
            "parking": int(data["parking"]),
            "elevador": int(data["elevador"]),
            "mobiliado": int(data["mobiliado"]),
            "piscina": int(data["piscina"]),
            "novo": int(data["novo"]),
            "municipio": data["municipio"],
            "tipo_negociacao": data["tipo_negociacao"],
            "tipo_imovel": data["tipo_imovel"],
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"])
        }])

        prediction = model.predict(input_df)
        return jsonify({"valor_previsto_aluguel": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)