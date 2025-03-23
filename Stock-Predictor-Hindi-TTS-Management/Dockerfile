# Use an official lightweight Python image.
FROM python:3.8-slim

# Install system-level dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --use-deprecated=legacy-resolver -r requirements.txt

# Copy the application code.
COPY run.py .

# Expose the port that FastAPI will run on.
EXPOSE 5050

# Run the application with Uvicorn.
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5050"]
