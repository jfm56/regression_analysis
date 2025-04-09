# Use an official Python runtime as the base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Command to run the application
# Install package in development mode
RUN pip install -e .

CMD ["python", "-m", "regression_analysis.regression"]
