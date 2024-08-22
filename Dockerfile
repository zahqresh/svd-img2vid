# Include Python
FROM python:3.11.1-buster

# Define your working directory
WORKDIR /

# Install all libs from requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install runpod
RUN pip install runpod
RUN pip install torch
RUN pip install uuid
RUN pip install boto3
RUN pip install requests
RUN pip install diffusers
RUN pip install transformers
RUN pip install accelerate>=0.17.0
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Add your file
ADD img2vid.py .

# Call your file when your container starts
CMD [ "python", "-u", "/img2vid.py" ]
