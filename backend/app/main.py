from flask import Flask
from flask_cors import CORS
from app.model.ner_model_provider.model_registry import model_registry
from app.utils.load_models import init_store_models
from service.app_router import api


if __name__ == '__main__':
    # int the store and load the base models
    models= init_store_models()
    for model in models:
        model_registry.add_model(model)

    # start the rest api
    app = Flask(__name__)
    app.register_blueprint(api)
    CORS(app)

    @app.after_request
    def add_cache_control_headers(response):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    app.run(host='0.0.0.0',debug=False)


