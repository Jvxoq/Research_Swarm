# Dockerfile
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first (for layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app directory - Copy the App code
COPY ./app /app/app

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
