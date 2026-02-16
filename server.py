import math
import multiprocessing
import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from engine import WanInferenceEngine
import time

# --- API Key Security ---
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return key  # No key configured, allow all requests
    if key != API_KEY:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return key

# --- Constants ---
FPS = 24

def seconds_to_frames(seconds: float) -> int:
    """Convert seconds to the nearest valid Wan frame count (must be 4k+1)."""
    raw_frames = seconds * FPS
    k = max(1, math.ceil((raw_frames - 1) / 4))
    return 4 * k + 1

# --- Pydantic Models for Validation ---
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, distorted, deformation, dull"
    width: int = 1280
    height: int = 720
    seconds: float = 3.375  # ~81 frames at 24fps
    steps: int = 4  # Default to LightX2V setting

class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int

# --- Worker Function ---
def gpu_worker(device_id, queue, status_dict, output_dir, model_path):
    # Set CUDA_VISIBLE_DEVICES to ensure isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # Initialize Engine (Expensive operation, done once)
    engine = WanInferenceEngine(0, model_path) # Always 0 inside the isolated process
    
    while True:
        job = queue.get()
        if job is None: break # Poison pill to stop worker
        
        job_id, params = job
        status_dict[job_id] = "processing"
        
        try:
            print(f"[GPU {device_id}] Starting Job {job_id}")
            start_time = time.time()
            
            frames = seconds_to_frames(params.seconds)
            video_frames = engine.generate(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                frames=frames,
                steps=params.steps
            )

            # Save Video
            save_path = os.path.join(output_dir, f"{job_id}.mp4")
            from diffusers.utils import export_to_video
            export_to_video(video_frames, save_path, fps=FPS)
            
            duration = time.time() - start_time
            status_dict[job_id] = {"status": "completed", "path": save_path, "duration": duration}
            print(f"[GPU {device_id}] Job {job_id} Finished in {duration:.2f}s")
            
        except Exception as e:
            print(f"[GPU {device_id}] Error: {e}")
            status_dict[job_id] = {"status": "failed", "error": str(e)}

# --- Main API ---
app = FastAPI(
    title="Wan 2.2 Blackwell Service",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global State
job_queue = multiprocessing.Queue()
manager = multiprocessing.Manager()
job_status = manager.dict()
OUTPUT_DIR = "/workspace/outputs"
MODEL_PATH = "/workspace/models/Wan2.2-14B"

@app.on_event("startup")
def startup_event():
    import torch
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("WARNING: No GPUs detected. No workers will be started.")
        return

    print(f"Detected {gpu_count} GPU(s). Spawning workers...")
    for i in range(gpu_count):
        p = multiprocessing.Process(
            target=gpu_worker,
            args=(i, job_queue, job_status, OUTPUT_DIR, MODEL_PATH)
        )
        p.start()
        print(f"  Started worker on GPU {i}: {torch.cuda.get_device_name(i)}")

@app.post("/generate", response_model=JobResponse)
def generate(req: GenerationRequest, _key: str = Security(verify_api_key)):
    job_id = str(uuid.uuid4())
    job_status[job_id] = "queued"
    job_queue.put((job_id, req))
    return {"job_id": job_id, "status": "queued", "queue_position": job_queue.qsize()}

@app.get("/status/{job_id}")
def get_status(job_id: str, _key: str = Security(verify_api_key)):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status[job_id]

@app.get("/download/{job_id}")
def download_video(job_id: str, _key: str = Security(verify_api_key)):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    status = job_status[job_id]
    if isinstance(status, str) or status.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Video is not ready yet")
    video_path = status["path"]
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)