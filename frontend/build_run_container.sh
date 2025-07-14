docker build -t oralner-website:v1 .
docker run -d -p 8080:80 oralner-website:v1