import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

# Configuration
# -------------------------------------------------------------------------
# We use a shared volume path as defined in the Docker strategy
MODEL_ROOT = os.getenv("MODEL_ROOT", "/workspace/models")

# 1. The Official Base Model (Diffusers Format)
# Contains: T5 Encoder, VAE, Tokenizer, and Scheduler configs
OFFICIAL_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B"
OFFICIAL_DIR = os.path.join(MODEL_ROOT, "Wan2.2-T2V-A14B")

# 2. The Distilled Model (LightX2V - 4 Steps)
# Contains: Optimized DiT weights in FP8 (e4m3fn)
LIGHTX2V_REPO_ID = "lightx2v/Wan2.2-Distill-Models"
LIGHTX2V_FILENAME = "wan2.2_t2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
LIGHTX2V_DIR = os.path.join(MODEL_ROOT, "LightX2V")

def download_official_base():
    """
    Downloads the full official repository.
    This ensures we have the correct folder structure (vae/, text_encoder/, etc.)
    required by the WanPipeline.
    """
    print(f"[-] Downloading Official Base Model: {OFFICIAL_REPO_ID}...")
    try:
        snapshot_download(
            repo_id=OFFICIAL_REPO_ID,
            local_dir=OFFICIAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            # We exclude the massive original FP32 transformer if we only plan to use LightX2V
            # However, for safety/fallback, we often download everything. 
            # Uncomment the line below to save ~30GB if you ONLY want LightX2V:
            # ignore_patterns=["transformer/*.pth", "transformer/*.safetensors"]
        )
        print(f"[+] Official model downloaded to: {OFFICIAL_DIR}")
    except Exception as e:
        print(f"[!] Error downloading official model: {e}")
        sys.exit(1)

def download_lightx2v_distill():
    """
    Downloads the specific 4-step distilled FP8 checkpoint.
    This replaces the standard transformer to enable ~6s generation speed.
    """
    print(f"[-] Downloading LightX2V Distilled Weights (FP8)...")
    try:
        hf_hub_download(
            repo_id=LIGHTX2V_REPO_ID,
            filename=LIGHTX2V_FILENAME,
            local_dir=LIGHTX2V_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[+] LightX2V weights downloaded to: {LIGHTX2V_DIR}/{LIGHTX2V_FILENAME}")
    except Exception as e:
        print(f"[!] Error downloading LightX2V weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Wan 2.2 Model Acquisition...")
    print(f"Target Directory: {MODEL_ROOT}")
    
    # Ensure directories exist
    os.makedirs(OFFICIAL_DIR, exist_ok=True)
    os.makedirs(LIGHTX2V_DIR, exist_ok=True)

    # Execute Downloads
    download_official_base()
    download_lightx2v_distill()
    
    print("\n All models downloaded.")
    print("To use these in the inference engine:")
    print(f"1. Load the pipeline from: {OFFICIAL_DIR}")
    print(f"2. Swap the transformer weights with: {LIGHTX2V_DIR}/{LIGHTX2V_FILENAME}")