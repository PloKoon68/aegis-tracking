# ✅ Slim base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install OpenCV dependency
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-dev \
 && rm -rf /var/lib/apt/lists/*
# ✅ Copy only what you need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy rest (done later to keep Docker cache)
COPY . .

# ✅ Run your app with production-ready server (optional)
#CMD ["gunicorn", "--bind", "0.0.0.0:4000", "app:app"]
CMD ["python", "app.py"]

