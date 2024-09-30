# Use an official Python runtime as a parent image
FROM python:3.12.4-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD wsgi.py /app
ADD embedding_models /app/embedding_models
ADD tools /app/tools
ADD .env /app
ADD requirements.txt /app

# Install ffmpeg
RUN apt update && apt full-upgrade -y
RUN apt install ffmpeg -y && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r requirements.txt

# Define volume to be mounted
VOLUME /app/DRIVE /app/backup /app/logs

# Make port available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=wsgi.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Start flask server
CMD ["flask", "run"]