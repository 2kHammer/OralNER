docker build -t oralner-website:v1 .
docker run -d -p 8080:80 -v ./config:/etc/nginx/config hammeralex2000/oralner-website:v10