from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('HOUSE.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    val1 = request.form['No Of Bedroom']
    val2 = request.form['No Of Bathroom']
    val3 = request.form['Area required in sqft']
    val4 = request.form['status']
    val5 = request.form['Location']
    val6 = request.form['facing']
    val7 = request.form['type']
    data = np.array([val1, val2, val3, val4, val5, val6, val7])
    data = data.astype(np.float64)
    print(data)
    pred = model.predict([data])[0]

    return render_template('home.html', prediction_text="The predicted House price was {:.2f}lakhs".format(pred))
app.run()
