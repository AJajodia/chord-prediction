from flask import Flask, request, render_template
import joblib
from pandas import DataFrame
from pydub import AudioSegment
import logging
import sys



model = joblib.load("random_forest.joblib")
encoder = joblib.load("encoder.joblib")

app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the request
    request_info = DataFrame({'first':request.form['first'], 'second':request.form['second'], 'third':request.form['third']}, index = [0])

    # Preprocess the input features
    # Make predictions using the loaded model
    prediction = [
        request.form['first'],
        request.form['second'],
        request.form['third'],
        model.predict(encoder.transform(request_info))[0]
        ]
    
    file_names = []

    for chord in prediction:
        if chord.islower():
            file_names.append(chord + '_m')
        else:
            file_names.append(chord)

    filename = "static/audio/progressions/progression" + "".join(file_names) + ".mp3"

    progression = AudioSegment.from_mp3("static/audio/" + file_names[0] + ".mp3")
    for chord in file_names[1:4]:
        progression = progression + AudioSegment.from_mp3("static/audio/" + chord + ".mp3")
    
   
    progression.speedup().export(filename, format="mp3")

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=False)