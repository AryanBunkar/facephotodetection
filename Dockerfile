# Use official Python image
FROM python:3.10-slim

# Install system dependencies (needed for dlib & OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy your Python script and any required files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir numpy opencv-python face_recognition

# Run your main script
CMD ["python", "facephoto.py"]
