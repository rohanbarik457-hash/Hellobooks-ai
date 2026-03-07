# Use Python 3.10
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install all required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "app.py"]
