FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/app ./app

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
