# Step 1: Use an official Python base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /src

# Step 3: Install the necessary dependencies
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

# Step 6: Define the command to run the application
CMD ["python", "app.py"]  
