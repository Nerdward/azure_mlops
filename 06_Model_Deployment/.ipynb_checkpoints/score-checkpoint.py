import json
import logging
import numpy as np
import os
import pickle
import joblib
import onnxruntime
import time

def init():
    global model, scaler, input_name, label_name
    # , inputs_dc, prediction_dc
    

    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model/scaler/scaler.pkl')
    # deserialize the model file back into a sklearn model
    scaler = joblib.load(scaler_path)
    
    model_onnx = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model/support-vector-classifier/svc.onnx')
    # print(os.listdir(model_onnx))
    model = onnxruntime.InferenceSession(model_onnx, None)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    
 
def run(data):
    try: 
        data = json.loads(data)["data"]
        data = np.array(data)
        data = scaler.fit_transform(data.reshape(1, 6))

        # model inference
        result = model.run([label_name], {input_name: data.astype(np.float32)})[0]
        # this call is saving model output data into Azure Blob

    except Exception as e:   
        result = 'error'
        print(e)

    return result.tolist()       
