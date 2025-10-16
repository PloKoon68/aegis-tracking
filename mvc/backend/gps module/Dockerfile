# ✅ Slim base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# ✅ Copy only what you need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy rest (done later to keep Docker cache)
COPY . .

# ✅ Run your app with production-ready server (optional)
#CMD ["gunicorn", "--bind", "0.0.0.0:4000", "app:app"]
CMD ["python", "app.py"]

