# Use an official Python runtime as a parent image
FROM python:3.10-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && apt-get clean
RUN apt-get update && apt-get install -y portaudio19-dev

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask API
EXPOSE 5001

# Run the Flask app
CMD ["python", "audio_service_app.py"]
