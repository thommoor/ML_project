from flask import Flask
from flask import jsonify
import connexion
from joblib import load

#load the model

#my_model = load('svc_model.pkl')

# Create the application instance
app = connexion.App(__name__, specification_dir="./")

# Read the yaml file to configure the endpoints
app.add_api("master.yaml")

# create a URL route in our application for "/"
@app.route("/")
def home():
    msg = """<html><head><b>Heart Failure Prediction Model:</b></head><body><p>Add "/prediction/" to the url in the search bar</p>
		<p>then add 12 values for Age, Anaemia, Creatinin Phosphokinase, Diabetes, Ejection Fraction, High Blood Pressure, High Blood Pressure, Platelets, Serum Creatinine, Serum Sodium, Sex, Smoking status, and Days between Follow-Up Visits.</p>
		<p> Separate each values with a coma (,).</p></body></html>"""
    return msg


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
