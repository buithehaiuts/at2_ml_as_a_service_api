# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt from the root of the project to the working directory
COPY requirements.txt ./  

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container, including subdirectories
COPY backend/ . 

# Expose the port the app runs on
EXPOSE 8000  # Expose port 8000 for the FastAPI app

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
