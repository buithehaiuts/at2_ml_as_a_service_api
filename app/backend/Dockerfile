# Use Python 3.11-slim as the base image (you can also stick with 3.9-slim if you prefer)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install the dependencies specified in pyproject.toml
RUN poetry install --no-interaction --no-root

# Copy the backend folder inside the app folder
COPY app/backend/ ./backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run FastAPI
CMD ["poetry", "run", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
