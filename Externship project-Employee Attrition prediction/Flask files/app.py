from flask import Flask, request, render_template
from joblib import load

import requests

import json
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "W4s61PF1oyxh6Q5EDbM_doFY4Wt2B90X7D_JGdojQXjp"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app=Flask(__name__)

column=load('onehot.save')
model=load('model.save')
scalar=load('scalar.save')

@app.route('/')
def home():
    return render_template("indexEA.html")

@app.route('/y_predict',methods=['POST'])
def y_predict():
    x_test=[[x for x in request.form.values()]] 
    print(x_test)
    x_test=column.transform(x_test) 
    print(x_test)
    x_test=scalar.transform(x_test)
    print(x_test)
    x_test=x_test.tolist()
    payload_scoring = {"input_data": [{"field": [["D1","D2","D3","D4","D5","D6","D7","S1","S2","satisfaction_level","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years"]], 
                                   "values": x_test}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/564494da-ff78-4df4-8fbc-02c1ff6c8cf2/predictions?version=2021-08-03', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions = response_scoring.json()
    pred = predictions['predictions'][0]['values'][0][0]
    if pred==0:
        output = 0
        print("Yay! A happy employee! We thank them for their service.")
    else:
        output = 1
        print("Sorry! This employee has decided to leave. We are sorry to see them go!")
    return render_template("resultEA.html",prediction_text=output)

if __name__=='__main__':
    app.run(debug=True)