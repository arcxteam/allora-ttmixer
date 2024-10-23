# Stage 1: Build environment
FROM python:3.9-slim AS project_env

RUN apt-get update && apt-get install -y curl

# Instal transformer 
RUN pip install transformers

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final environment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the build environment from the first stage
COPY --from=project_env /usr/local /usr/local

# Copy application files
COPY . /app/

# Expose the port (Gunicorn will use this)
EXPOSE 8000

# Set the entrypoint command to run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
