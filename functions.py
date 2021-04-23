from joblib import load
import numpy as np
import json
from flask import jsonify
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os

#load the model

my_model = load('svc_model.pkl')

#x_test = my_model.x_test
#y_test = my_model.y_test
#pred = my_model.predict(x_test)

def my_prediction(id):
	dum = np.array(id)
	dumT = dum.reshape(1,-1)
	dum_str = dum.tolist()
	r = dum.shape
	t = dumT.shape
	r_str = json.dumps(r)
	t_str = json.dumps(t)
	array = np.zeros(shape = (12,12))
	arr = dumT + array
	prediction = my_model.predict(arr)
	pred = prediction[0]
	if pred == 1:
		msg = "DEATH EVENT PREDICTED"
	else:
		msg = "DEATH EVENT NOT PREDICTED"
	msg1 = suggestions(id, msg)

	return msg1

def suggestions(arr, str):
	sug_str = [str]

	if arr[1] == 1:
		sug_str = np.append(sug_str, "Speak with your doctor about anaemia.")
	if arr[2] > 100:
		sug_str = np.append(sug_str, "Speak with your doctor about elevated CPK levels.")
	if arr[3] == 1:
		sug_str = np.append(sug_str, "Consider options to mitigate the effects of diabetes.")
	if arr[4] < 50:
		sug_str = np.append(sug_str, "Percentage of blood leaving the heart is low. Speak with doctor.")
	if arr[5] ==1:
		sug_str = np.append(sug_str, "Your heart is overworked.  Consider treatment for high blood pressure.")
	if arr[6] > 450000:
		sug_str = np.append(sug_str, "Platelet count high.  Speak with doctor.")
	if arr[6] < 150000:
		sug_str = np.append(sug_str, "Platelet count low.  Speak with doctor.")
	if arr[7] > 1.0:
		sug_str = np.append(sug_str, "Serum Creatinine at high levels.  Your doctor may have suggestions.")
	if arr[8] < 135:
		sug_str = np.append(sug_str, "Serum Sodium at low levels. Your doctor may consider medication.")
	if arr[9] == 0:
		sug_str = np.append(sug_str, "Women are more likely to suffer heart failure. Speak with your doctor about prevention.")
	if arr[10] == 1:
		sug_str = np.append(sug_str, "Cessation of smoking is strongly suggested.")
	sug = sug_str.tolist()
	return sug

def info():
	msg = [
		"Age: Your risk of heart failure begins to increase with age around 65.",
		"Anaemia: Anaemia causes your heart to work harder, so anemic persons are at higher risk of heart failure.",
		"Creatinin Phosphokinase: High level of the enzyme CPK is linked to heart failure.",
		"Diabetes: Diabetetes can easily lead to heart disease.",
		"Ejection Fraction: The percentage of blood leaving the heart at each contraction. Less than 50% leads to high risk.",
		"High Blood Pressure: High levels mean the heart is overworked.",
		"Platelets: Platelet counts should be between 150,000 and 450,000 /mL to be considered low risk.",
		"Serum Creatinine: High level means the kidneys aren't working well, leading to irregularities in blood flow and increasing the risk of heart failure.",
		"Serum Sodium: Level below 135 mEq/L is a factor in heart failure.",
		"Sex: Women are more likely to suffer heart failure than men."
		"Smoking: Smoking is generally bad for your health and can easily lead to heart failure among other ailments."
		]
	return msg

def prediction_test():
	msg = """<html><head><b>Input: user input here</b></head><body><p>Default Prediction: DEATH! Change your ways NOW!!!</p></body></html>"""
	return msg

def test(a, b):
	a = float(a)
	b = float(b)
	out = a + b
	return str(out)

def model_score():
	#report = (classification_report(y_test, pred, target_names=['0', '1']))
	message = "hello"
	#print(my_model.predict([0,0,0,0,0],[0,0,0,0,0,0]))
	message = "\nprecision\trecall\tf1-score\tsupport\n\n0\t0.84\t0.82\t0.83\t38\n1\t0.84\t0.86\t0.85\t44\n\naccuracy\t\t\t0.84\t82\nmacro avg\t0.84\t0.84\t0.84\t82\nweighted avg\t0.84\t0.84\t0.84\t82\n"
	return message