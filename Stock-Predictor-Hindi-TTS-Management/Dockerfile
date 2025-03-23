# Use an official lightweight Python image.
FROM python:3.8-slim

ENV NUMBA_CACHE_DIR=/tmp/numba_cache


# Install system-level dependencies.
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
 && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container.
WORKDIR /app

ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.config/matplotlib

RUN mkdir -p /tmp/.cache /tmp/.config/matplotlib /tmp/.local/share/tts_models \
    && chmod -R 777 /tmp/.cache /tmp/.config /tmp/.local

# Copy the requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --use-deprecated=legacy-resolver -r requirements.txt

# Copy the application code.
COPY run.py .

# Expose the port that FastAPI will run on.
EXPOSE 5050

# Run the application with Uvicorn.
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5050", "--workers", "1"]
