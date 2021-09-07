FROM node:12-alpine

WORKDIR /app/

COPY package*.json /app/

RUN npm install

COPY . /app/

ENV PORT=8080
ENV HOST=0.0.0.0
EXPOSE 8080

RUN npm run build

CMD ["npm", "start"]
