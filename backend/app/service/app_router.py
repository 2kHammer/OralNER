from flask import jsonify, Blueprint, request, abort

from app.controller import model_manager, data_manager, ner_manager

api = Blueprint('api', __name__)

@api.route('/models', methods=['GET'])
def get_models():
    try:
        return jsonify(model_manager.get_models()),200
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/models/<int:id>', methods=['GET'])
def get_model(id):
    try:
        model = model_manager.get_model(id)
        if model is None:
            abort(404, description="Model not found")
        return jsonify(model), 200
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/models/active', methods=['GET'])
def get_model_active():
    try:
        model = model_manager.get_model_active()
        return jsonify(model), 200
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/models/active/<int:id>', methods=['PUT'])
def set_model_active(id):
    try:
        if model_manager.set_model_active(id):
            return '',204
        else:
            abort(404, description="Model not found")
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))


@api.route('/trainingdata', methods=['GET'])
def get_training_data():
    try:
        return jsonify(data_manager.get_training_data()), 200
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/ner', methods=['POST'])
def apply_ner():
    try:
        if request.is_json:
            data = request.get_json()
            text = data['text']
            tokens, labels, metrics = ner_manager.ner(text, False)
            return jsonify([tokens,labels]), 200
        elif "file" in request.files:
            file = request.files["file"]
            decoded_file = file.read().decode("utf-8").splitlines()
            tokens, labels, metrics = ner_manager.ner(decoded_file, True)
            return jsonify([tokens,labels,metrics]), 200
        return jsonify({'error': 'No valid text or file provided'}), 400
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))


@api.route('/ner/finetune', methods=['POST'])
def finetune():
    try:
        if request.is_json:
            data = request.get_json()
            model_id = data.get("model_id")
            dataset_id = data.get("dataset_id")
            parameters = data.get("parameters")
            name = parameters.get("new_model_name")
            if model_id is None or dataset_id is None or name is None:
                return jsonify({'error': 'No valid model_id or dataset_id or parameters provided'}), 400
            model_id = ner_manager.finetune_ner(model_id, dataset_id, name)
            return jsonify({"modified_model_id": model_id}), 202

    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/trainingdata', methods=['POST'])
def add_trainingsdata():
    try:
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename
            dataset_name = request.form.get("dataset_name","NoDatasetNameProvided")
            decoded_file = file.read().decode("utf-8").splitlines()
            add_data_ok =data_manager.add_training_data(dataset_name, filename,decoded_file)
            if add_data_ok:
                return '',201
            else:
                return jsonify({'error': 'no valid dataset format'}), 400
        else:
            return jsonify({'error': 'No file provided'}), 400
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))
