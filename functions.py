from joblib import load
import numpy as np
import json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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
	#return pred

	pred_str = pred.tolist()
	return pred_str

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

def matrix():
	pred = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
 0, 0, 1, 1, 0, 1, 0, 0]
	test = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
 1, 0, 1, 1, 0, 1, 0, 0]
	conf = confusion_matrix(test, pred)
	message = conf[0] + conf[1]
	print(list(conf))
	msg = {conf}
	return msg

def roc():
	#probs = pipe.predict_proba(x_test)
	#preds = probs[:,1]
	#fpr, tpr, threshold = metrics.roc_curve(y_test, dockerpred)
	#roc_auc = metrics.auc(fpr, tpr)

	#plt.title('Receiver Operating Characteristic')
	#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	#plt.legend(loc = 'lower right')
	#plt.plot([0, 1], [0, 1],'r--')
	#plt.xlim([0, 1])
	#plt.ylim([0, 1])
	#plt.ylabel('True Positive Rate')
	#plt.xlabel('False Positive Rate')
	#plt.show()
	#return
	pass