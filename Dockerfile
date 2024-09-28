# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements files for both backend and frontend
COPY ./app/requirements.backend.txt ./requirements.backend.txt  # Adjust if you have separate requirements
COPY ./app/requirements.frontend.txt ./requirements.frontend.txt

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements.backend.txt

# Copy the backend application code to the container
COPY ./app/backend /app/backend  # Copy the backend directory

# Expose the port for FastAPI
EXPOSE 8000

# Start FastAPI in the background
CMD ["uvicorn", "backend/api:app", "--host", "0.0.0.0", "--port", "8000"] &

# Install frontend dependencies
RUN pip install --no-cache-dir -r requirements.frontend.txt

# Copy the frontend application code to the container
COPY ./app/frontend /app/frontend  # Copy the frontend directory

# Expose the port for Streamlit
EXPOSE 8501

# Command to start the Streamlit app
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
