import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import helper
from helper import predict_from_sentenca, preprocessar_sentenca

app = Flask(__name__)
CORS(app)

# Carregando o Modelo (os scores)

print('Carregando o modelo')
model = pickle.load(open('model.pkl','rb'))

@app.route('/sentiment_analysis', methods=['POST'])

def predict():
    
    print('Recebendo os dados via POST')
    data = request.get_json(force=True)
    print(data)
    
    sentenca = data['text']
    print('Sentenca: ', sentenca)

    print('Realizando a predicao')
    prediction = predict_from_sentenca(sentenca, model, preprocessar_sentenca)
    

    print('Predicao: ', prediction)
    print('Respondendo a requisicao')

    output = jsonify(prediction)
    print(output)

    return output

if __name__ == '__main__':
    app.run(port=5000, debug=True)

