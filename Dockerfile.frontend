# Use a Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app/frontend

# Copy the requirements file for the frontend to the working directory
COPY ./requirements.txt ./  # Ensure this includes the Streamlit dependencies

# Install frontend dependencies
RUN pip install --no-cache-dir -r requirements.txt  # Install the dependencies

# Copy the frontend application code to the container
COPY ./app/frontend /app/frontend  # Copy only the frontend directory

# Expose the port for Streamlit
EXPOSE 8501

# Command to start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]  # Adjust as necessary based on your entry point
