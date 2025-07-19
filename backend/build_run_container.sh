docker build -t oralner:v10 .
docker run -p 5000:5000 -v /home/alex/SS25/CBIE/OralNER/backend/test_mount:/backend/app/store/NER-Models/modified oralner:v10