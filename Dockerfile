# Use a Python base image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt 

# Copy the entire 'app' folder from the host to the '/app' folder in the container
COPY ./app /app

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "/app/app.py"]
