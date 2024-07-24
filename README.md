# Chord Prediction 
A project using machine learning to predict the final chords of user-selected chord progressions. Packages used: Pandas, Scikit-Learn, Flask, Pydub, Joblib, NumPy

Deployed via Heroku at https://chord-prediction-53c13a7a8745.herokuapp.com/

## Model
Using Scikit-Learn's OrdinalEncoder, chords are encoded as integers and processed as NumPy arrays. They are then passed to the model, a Scikit-Learn Random Forest Classifier with about 90% accuracy. The model suggests the most apt chord for the given progression and returns it as a string.

## Backend
The backend is written in Flask. It uses a simple endpoint system, receiving the inputs through a POST, passing them through the model, and finally returning all four chords as a progression and audio file. The model and encoder are called from Joblib files. After the progression is calculated, Pydub adds the proper files together and saves them as a single progression, which is passed to the POST output.

## Frontend
The frontend is an HTML form with minimal CSS.
