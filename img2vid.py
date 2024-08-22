import os
import requests
import torch
import uuid
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import boto3
import runpod
# Load the model outside the handler function to ensure it loads only once
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

def handler(job):
    job_input = job.get("input", {})

    # Validate input
    url = job_input.get("url")
    aws_access_key_id = job_input.get("aws_id")
    aws_secret_access_key = job_input.get("aws_secret")
    bucket_name = job_input.get("s3_bucket")

    if not url or not aws_access_key_id or not aws_secret_access_key or not bucket_name:
        return {
            "error": "Missing required input keys: 'url', 'aws_id', 'aws_secre', 's3_bucket'."
        }

    # Initialize boto3 session
    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Create an S3 client
    s3_client = boto3_session.client('s3')

    # Define the output directory and ensure it exists
    output_dir = os.path.abspath('videos')
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    video_filename = f"{unique_id}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    try:
        # Load the image
        image = load_image(url)
        image = image.resize((1024, 576))

        # Generate frames
        generator = torch.manual_seed(42)
        frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]

        # Export the video
        export_to_video(frames, video_path, fps=7)

        # Upload the video to S3
        s3_key = f"videos/{video_filename}"
        s3_client.upload_file(video_path, bucket_name, s3_key)

        # Generate a presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # Set expiration to 1 hour
        )

        return {"status": "success", "video_url": presigned_url}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Check if the video file exists before attempting to delete it
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Successfully deleted '{video_filename}' from the 'videos' folder.")
            except PermissionError:
                print(f"Failed to delete '{video_filename}'. The file might still be in use.")
        else:
            print(f"The video file '{video_filename}' was not found, so it could not be deleted.")

# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
