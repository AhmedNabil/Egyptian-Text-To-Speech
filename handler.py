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

# Global model variable
model = None
config = None

def load_model():
    """Load EGTTS model"""
    global model, config
    
    print("🚀 Loading EGTTS V0.1 model...")
    
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
    
    print("✅ Model loaded successfully!")

def handler(job):
    """
    RunPod handler for EGTTS
    
    Input format:
    {
        "text": "ازيك يا معلم",
        "language": "ar",
        "temperature": 0.75
    }
    """
    global model, config
    
    # Load model if not loaded
    if model is None:
        load_model()
    
    job_input = job.get("input", {})
    
    text = job_input.get("text", "")
    language = job_input.get("language", "ar")
    temperature = job_input.get("temperature", 0.75)
    
    if not text:
        return {"error": "No text provided"}
    
    try:
        print(f"📝 Generating speech for: {text}")
        
        # Get default conditioning (no speaker audio)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=None
        )
        
        # Generate speech
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
        
        print("✅ Speech generated successfully!")
        
        return {
            "audio": audio_b64,
            "sample_rate": 24000,
            "text": text
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}

# Start the serverless handler
if __name__ == "__main__":
    print("🎙️ Starting EGTTS Serverless Handler...")
    runpod.serverless.start({"handler": handler})