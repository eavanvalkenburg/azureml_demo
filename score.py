
import pickle
import json
import numpy
from azureml.core.model import Model

def init():
    global model
    from sklearn.externals import joblib

    # load the model from file into a global object
    model_path = Model.get_model_path(model_name="house_prices_regression")
    model = joblib.load(model_path)


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
