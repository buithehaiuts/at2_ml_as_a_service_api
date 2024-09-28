# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./  # Copy the requirements file to install dependencies

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt  # Install the dependencies

# Copy the application code to the container
COPY ./app /app  # Copy the entire app directory (both backend and frontend)

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000  # For FastAPI
EXPOSE 8501  # For Streamlit

# Use a CMD instruction to start the FastAPI app or Streamlit app
# CMD ["uvicorn", "backend/api.py:app", "--host", "0.0.0.0", "--port", "8000"]  # Uncomment to run FastAPI
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]  # Uncomment to run Streamlit
