# slim Python base image
FROM python:3.11-slim

WORKDIR /backend
#install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest, besides the store
RUN mkdir ./app
COPY app/controller/ ./app/controller/
COPY app/model/ ./app/model/
COPY app/service ./app/service/
COPY app/utils ./app/utils/
RUN mkdir -p ./app/store ./app/store/NER-Models ./app/store/NER-Models/modified
COPY app/store/Trainingsdata/ ./app/store/Trainingsdata/
COPY app/main.py ./app/main.py
COPY app/__init__.py ./app/__init__.py

# copy base models
# how to handle the models
# the base models have to be included in the image or a central hub 
# -> if an python script downloads them -> Container starts last to long
# if i would use nfs -> model have to be downloaded locally too
COPY app/store/NER-Models/base/ ./app/store/NER-Models/base/
COPY app/store/NER-Models/base_models_metadata.json ./app/store/NER-Models/models_metadata.json

ENV PYTHONPATH=/backend

EXPOSE 5000

CMD ["python","app/main.py"]


