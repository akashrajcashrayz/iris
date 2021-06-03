from flask import Flask, request, render_template
import numpy as np
import pickle
import joblib

# Create Flask object to run

app = Flask(__name__,template_folder= 'templates' )


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	knnIrisModel = joblib.load('irismodell.pkl')
	int_features = [float(x) for x in request.form.values()]


	final_features = [np.array(int_features)]

	prediction = knnIrisModel.predict(final_features)
	output =prediction[0]
	return render_template('index.html', prediction_text= output)

    
if __name__ == "__main__":
	app.run()
