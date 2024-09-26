# Use a Python base image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (not just app) to the container
COPY . /app

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the default command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
