# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY ./app/backend /app/backend  # Copy the backend directory
COPY ./app/frontend /app/frontend  # Copy the frontend directory

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000  # For FastAPI
EXPOSE 8501  # For Streamlit

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn backend/api:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
