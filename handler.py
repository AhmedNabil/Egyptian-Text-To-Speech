import runpod
import torch
import torchaudio
import io
import base64
import os

# Set environment
os.environ['COQUI_TOS_AGREED'] = '1'

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Global variables
model = None
config = None

def load_model():
    """Load EGTTS model once"""
    global model, config
    
    if model is not None:
        return
    
    print("🚀 Loading EGTTS V0.1...")
    
    config = XttsConfig()
    config.load_json("/models/EGTTS-V0.1/config.json")
    
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir="/models/EGTTS-V0.1/",
        vocab_path="/models/EGTTS-V0.1/vocab.json",
        use_deepspeed=True
    )
    model.cuda()
    
    print("✅ Model loaded!")

def handler(job):
    """Handler for RunPod"""
    
    # Load model
    load_model()
    
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    language = job_input.get("language", "ar")
    temperature = job_input.get("temperature", 0.75)
    
    if not text:
        return {"error": "No text provided"}
    
    try:
        print(f"📝 Text: {text}")
        
        # Get conditioning
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=None
        )
        
        # Generate
        out = model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature
        )
        
        # Convert to base64
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            torch.tensor(out["wav"]).unsqueeze(0),
            24000,
            format="wav"
        )
        
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "audio": audio_b64,
            "sample_rate": 24000
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})