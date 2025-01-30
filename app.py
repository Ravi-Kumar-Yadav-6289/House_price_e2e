import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, url_for, render_template


app = Flask(__name__)
model = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main_page.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']