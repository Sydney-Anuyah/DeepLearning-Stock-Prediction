from flask import Flask
from flask import request, jsonify
from CNN import CNN
from LSTM import RNN
from MultilayerPerceptron import MultilayerPerceptron

app = Flask(__name__)

@app.route('/cnn', methods=['POST'])
def get_prediction_cnn():
    day = request.json['day']
    
    prediction = CNN(day)
    return jsonify({'message': 'Prediction Successful', 'accuracy': prediction}), 201

@app.route('/rnn', methods=['POST'])
def get_prediction_rnn():
    day = request.json['day']
    
    prediction = RNN(day)
    return jsonify({'message': 'Prediction Successful', 'accuracy': prediction}), 201

@app.route('/mpp', methods=['POST'])
def get_prediction_multiLayerPerceptron():
    day = request.json['day']
    
    prediction = MultilayerPerceptron(day)
    return jsonify({'message': 'Prediction Successful', 'accuracy': prediction}), 201

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=8080)
