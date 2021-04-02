from joblib import load
import numpy as np
import json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#load the model

my_model = load('svc_model.pkl')
x_test = my_model.x_test
y_test = my_model.y_test
pred = my_model.predict(x_test)

def my_prediction(an_array):
	prediction = my_model.predict(an_array)
	name = class_names[prediction]
	return name

def model_score(y_test, pred):
	report = (classification_report(y_test, pred, target_names=['0', '1']))
	return report

def matrix(y_test, pred):
	conf = confusion_matrix(y_test, pred)
	return conf

def roc(x_test, y_test, pred):
	probs = pipe.predict_proba(x_test)
	preds = probs[:,1]
	fpr, tpr, threshold = metrics.roc_curve(y_test, dockerpred)
	roc_auc = metrics.auc(fpr, tpr)

	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()