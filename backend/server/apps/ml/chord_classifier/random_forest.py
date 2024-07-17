# file backend/server/apps/ml/chord_classifier/random_forest.py
import joblib
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = '../../research/'
        self.model = joblib.load(path_to_artifacts + 'random_forest.joblib')
        self.encoder = joblib.load(path_to_artifacts + 'encoder.joblib')

    def preprocessing(self, input_data):
        input_data = pd.DataFrame(input_data, index=[0])
        input_data = self.encoder.transform(input_data)
        return input_data
    
    def predict(self, input_data):
        return self.model.predict(input_data)
    
    def postprocessing(self, input_data):
        return {'chord':input_data, 'status':'OK'}
    
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
        

        
