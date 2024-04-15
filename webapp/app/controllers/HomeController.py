from flask import render_template, jsonify
from app.services.predict import *
from app.services.rateModal import *
import joblib

def home():
    return render_template('index.html', **locals())

def home_predict(data):
    predict = predict_price(
        int(data['type']), 
        int(data['area']), 
        int(data['bedrooms']), 
        int(data['bathrooms']), 
        data['location']
    )

    result = {
        "result": predict
    }

    return jsonify(result)

def home_propose(data):
    predict = propose(
        int(data['type']), 
        int(data['area']), 
        int(data['bedrooms']), 
        int(data['bathrooms']), 
        data['location'],
        int(data['price'])
    )

    result = {
        "result": predict
    }

    return jsonify(result)
def home_rateModal():
    rate_Modal = rateModal()
    result = {
        "result": rate_Modal
    }
    return jsonify(result)

def notfound():
    return render_template('notfound.html')