docker build -t oralner:v10 .
docker run -p 5000:5000 -v /home/alex/SS25/CBIE/OralNER/backend/test_mount:/backend/app/store/NER-Models/modified oralner:v10

docker run -p 5000:5000 -v /home/alex/Nextcloud3/Alex/Uni/Master/SoSe25/CBIE/Experimente/ModModels:/backend/app/store/NER-Models/modified -v /home/alex/Nextcloud3/Alex/Uni/Master/SoSe25/CBIE/Experimente/base_models_metadata.json:/backend/app/store/NER-Models/models_metadata.json hammeralex2000/oralner:v13