# Use an official Python runtime as a parent image
FROM python:3.10-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 6000 for the AI service
EXPOSE 6000

# Run the AI service
CMD ["python", "ai_service_app.py"]
