
# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Set environment variables for the container
ENV PYTHONUNBUFFERED=TRUE

# Create and set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Set the entrypoint to run your main Python script
ENTRYPOINT ["python", "sentiment_model.py"]
