# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install the dependencies specified in pyproject.toml
RUN poetry install --no-interaction --no-root

# Copy the frontend folder containing app.py into the container
COPY app/frontend/ ./frontend

# Expose the port the app runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["poetry", "run", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
