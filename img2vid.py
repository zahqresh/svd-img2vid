import os
import requests
import torch
import uuid
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import runpod


def handler(job):
    job_input = job.get("input", {})
    
    # Validate input
    url = job_input.get("url")
    
    if not url:
        return {
            "error": "Input is missing the 'url' key."
        }
    
    # Define the output directory and ensure it exists
    output_dir = 'videos'
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    video_filename = f"{unique_id}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    try:
        # Load the image
        image = load_image(url)
        image = image.resize((1024, 576))

        # Initialize the pipeline
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline.enable_model_cpu_offload()

        # Generate frames
        generator = torch.manual_seed(42)
        frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]

        # Export the video
        export_to_video(frames, video_path, fps=7)

        # Upload the video to File.io
        api_endpoint = "https://file.io"
        with open(video_path, "rb") as video_file:
            response = requests.post(api_endpoint, files={"file": video_file})
            response.raise_for_status()  # Raise an exception for any unsuccessful status codes
            return response.json()  # Return the response as JSON

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.remove(video_path)
            print(f"Successfully deleted '{video_filename}' from the 'videos' folder.")
        except PermissionError:
            print(f"Failed to delete '{video_filename}'. The file might still be in use.")


# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
