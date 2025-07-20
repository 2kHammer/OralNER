FROM nginx:alpine

COPY src/ /usr/share/nginx/html/

# Replace nginx conf
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80