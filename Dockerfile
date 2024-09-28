# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY ./app /app

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run both the FastAPI app and the Streamlit app
CMD ["sh", "-c", "uvicorn backend.api:SalesAPI().app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
