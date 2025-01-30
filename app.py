import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, url_for, render_template


app = Flask(__name__)
# model loading
model = pickle.load(open('regmodel.pkl', 'rb'))
#scaler loading
scaler = pickle.load(open('scaling.pkl','rb'))

# could have been done with sklearn pipe and column transformers


@app.route('/')
def home():
    return render_template('main_page.html')


# this was for api test

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # 1,-1 because -1 adopts the number of columns available in the original list while making the rows to be one hence this states that all the json records turn out to be a single row of a dataset.
    data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(data)
    print(output[0])
    return jsonify(output[0])



# this is for the actual web app

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("main_page.html",prediction_text="The House price prediction is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)

