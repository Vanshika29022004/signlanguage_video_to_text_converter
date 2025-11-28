# Sign Language Converter - Docker Deployment Guide

This guide explains how to deploy the Sign Language to Text Converter application using Docker.

## Prerequisites

- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose (usually included with Docker Desktop)
- At least 4GB of RAM available
- The pre-trained model file (`models/sign_model.h5`) should be in the models folder

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run the Application

```bash
docker-compose up
```

The application will start on `http://localhost:5000`

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

## Stopping the Application

```bash
docker-compose down
```

## Building Without Docker Compose

If you prefer to use Docker directly:

### Build the image:
```bash
docker build -t sign-language-converter:latest .
```

### Run the container:
```bash
docker run -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  sign-language-converter:latest
```

## Common Commands

### View running containers:
```bash
docker ps
```

### View logs:
```bash
docker-compose logs -f sign-language-converter
```

### Rebuild after code changes:
```bash
docker-compose up --build
```

### Remove all containers and images:
```bash
docker-compose down -v
docker image rm sign-language-converter:latest
```

## Configuration

### Memory Settings

The docker-compose.yml includes memory limits. Adjust these if needed:

```yaml
deploy:
  resources:
    limits:
      memory: 4G         # Maximum memory
    reservations:
      memory: 2G        # Minimum memory
```

### Port Mapping

Change the port by modifying the `ports` section in docker-compose.yml:

```yaml
ports:
  - "8080:5000"  # Access at localhost:8080
```

### Environment Variables

Add environment variables to the `environment` section in docker-compose.yml or create a `.env` file.

## Troubleshooting

### Container exits immediately
- Check logs: `docker-compose logs`
- Ensure the model file exists: `models/sign_model.h5`

### Port already in use
- Change the port mapping in docker-compose.yml
- Or stop other services using port 5000

### Out of memory errors
- Increase memory limits in docker-compose.yml
- Reduce the number of workers in the Dockerfile

### Video processing is slow
- Increase allocated memory
- Reduce video resolution before uploading
- Use a GPU-enabled image (requires nvidia-docker)

## Production Deployment

For production use:

1. Use a reverse proxy (Nginx, Apache)
2. Enable HTTPS/SSL
3. Use environment variables for secrets
4. Implement proper logging and monitoring
5. Use health checks
6. Consider using a managed container service (AWS ECS, Google Cloud Run, etc.)

## GPU Support (Optional)

To use GPU acceleration for TensorFlow:

1. Install nvidia-docker
2. Replace the base image in Dockerfile:
```dockerfile
FROM tensorflow/tensorflow:latest-gpu
```

3. Update docker-compose.yml:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

## Performance Tips

- Pre-warm the model after startup
- Implement request caching for repeated predictions
- Use a CDN for static files
- Implement batch processing for multiple videos
- Monitor resource usage with `docker stats`

## Support

For issues or questions, check:
- Docker logs: `docker-compose logs -f`
- Flask error messages in the UI
- Application logs in the container
