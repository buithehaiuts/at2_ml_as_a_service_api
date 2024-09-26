# Use a Python base image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the entire 'app' folder from the host to the '/app' folder in the container
COPY ./app /app

# Expose the ports that FastAPI and Streamlit will run on
EXPOSE 8000 8501

# Set the command to run both the backend and frontend
CMD ["bash", "-c", "uvicorn app.backend.api:app --host 0.0.0.0 --port 8000 & streamlit run /app/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
