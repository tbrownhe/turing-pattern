# Stage 1: Build the frontend
FROM node:20 AS builder
WORKDIR /app

# Install dependencies and build
COPY package*.json /app/

RUN npm ci

COPY ./ /app/

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine 

# Copy built files into container
COPY --from=builder /app/dist /usr/share/nginx/html
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
