# Use a minimal image with Python and necessary libraries pre-installed
FROM python:3.11-slim

# Set environment variable for port
ENV PORT 8080
ENV PYTHONUNBUFFERED True

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
WORKDIR /app
COPY . /app

# Command to run the application (using Gunicorn for production)
# The default 30s Gunicorn timeout is fine for a worker, as Cloud Run will handle the final termination
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 3600 app:app