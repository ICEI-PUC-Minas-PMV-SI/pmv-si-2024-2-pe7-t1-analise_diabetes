from flask import Flask, request
from flask_cors import CORS
import pickle

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

def load_model(model_name):
    path = f"./{model_name}_model.pkl"
    model = None
    with open(path, 'rb') as file:
        model = pickle.load(file)
    if model is None:
        raise Exception(f"Model {model_name} not found")
    return model


naive_model = load_model('naive')
tree_model = load_model('tree')
forest_model = load_model('random')

fields = [
  'polyuria',
  'polydipsia',
  'age',
  'gender',
  'sudden_weight_loss',
  'partial_paresis',
  'polyphagia',
  'irritability',
  'alopecia',
  'visual_blurring',
  'weakness',
  'muscle_stiffness',
  'genital_thrush',
  'obesity',
  'delayed_healing',
  'itching',
]


@app.route('/', methods=['POST'])
def index():
    input = request.get_json()
    
    # validate if input have all fields with numeric values
    for key in fields:
        if key not in input:
            return {'error': f"Field {key} not found"}, 400
        if not isinstance(input[key], (int, float)):
            return {'error': f"Field {key} must be a number"}, 400
        if input[key] < 0:
            return {'error': f"Field {key} must be a positive number"}, 400
        if key == 'gender' and input[key] not in [1, 2]:
            return {'error': f"Field {key} must be 1 or 2"}, 400
        
    # convert to list in the same order of the model
    input_list = [input[key] for key in fields]

    # predict
    naive_pred = naive_model.predict([input_list])
    tree_pred = tree_model.predict([input_list])
    forest_pred = forest_model.predict([input_list])

    naive_result = bool(naive_pred[0])
    tree_result = bool(tree_pred[0])
    forest_result = bool(forest_pred[0])

    points = int(naive_result) + int(tree_result) + int(forest_result)

    final_classification = False if points < 2 else True

    return {
        'prevision': {
            'naive': naive_result,
            'tree': tree_result,
            'forest': forest_result,
        },
        'input': input_list,
        'final_classification': final_classification
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)