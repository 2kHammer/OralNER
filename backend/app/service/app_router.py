from flask import jsonify, Blueprint, request, abort

from app.controller import model_manager, data_manager

api = Blueprint('api', __name__)

@api.route('/models', methods=['GET'])
def get_models():
    try:
        return jsonify(model_manager.get_models()),200
    except Exception as e:
        abort(500, description="Internal Server Error: " + str(e))

@api.route('/models/<int:model_id>', methods=['GET'])
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

@api.route('/models/active/{id}', methods=['PUT'])
def set_model_active(id):
    try:
        if model_manager.set_model_active(id):
            return 204
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