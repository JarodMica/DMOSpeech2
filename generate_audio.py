#!/usr/bin/env python3
"""
Minimal inference script for DMOSpeech2 audio generation.
Usage: python generate_audio.py --text "Your text here" --ref_audio path/to/reference.wav --output output.wav
"""

import os
import sys
import argparse
import torchaudio
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "dmospeech2"))

from huggingface_hub import login
from dmospeech2.infer import DMOInference
import torch

def main():
    text = "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects."
    ref_text = "Some call me nature, others call me mother nature."
    ref_audio = "dmospeech2/f5_tts/infer/examples/basic/basic_ref_en.wav"
    # ref_audio = "2.wav"
    # ref_text = "Kokoro is an open-weight TTS model with 82 million parameters."
    parser = argparse.ArgumentParser(description="Generate audio using DMOSpeech2")
    parser.add_argument("--text", default=text, help="Text to synthesize")
    parser.add_argument("--ref_audio", default=ref_audio, 
                       help="Reference audio file path")
    parser.add_argument("--ref_text", default=ref_text,
                       help="Reference audio transcript")
    parser.add_argument("--output", default="generated_audio.wav", help="Output audio file path")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--teacher_steps", type=int, default=0, help="Number of teacher steps")
    parser.add_argument("--student_start_step", type=int, default=0, help="Student start step")
    parser.add_argument("--temperature", type=float, default=0.0, help="Duration sampling temperature")
    
    args = parser.parse_args()
    
    # Check if checkpoints exist
    student_checkpoint = "ckpts/model_85000.pt"
    duration_checkpoint = "ckpts/model_1500.pt"
    
    if not os.path.exists(student_checkpoint):
        print(f"Error: Student checkpoint not found at {student_checkpoint}")
        return 1
    
    if not os.path.exists(duration_checkpoint):
        print(f"Error: Duration checkpoint not found at {duration_checkpoint}")
        return 1
        
    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio not found at {args.ref_audio}")
        return 1
    
    # Initialize model
    try:
        tts = DMOInference(
            student_checkpoint_path=student_checkpoint,
            duration_predictor_path=duration_checkpoint,
            device=args.device,
            model_type="F5TTS_Base"
        )
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Generate audio
    import time
    while True:
        input("Press Enter to generate audio...")
        try:
            start_time = time.time()
            generated_audio = tts.generate(
                gen_text=args.text,
                audio_path=args.ref_audio,
                prompt_text=args.ref_text,
                teacher_steps=args.teacher_steps,
                duration=2000,
                student_start_step=args.student_start_step,
                temperature=args.temperature,
                cfg_strength=4.0
            )
            end_time = time.time()
            
            # Calculate RTF
            processing_time = end_time - start_time
            audio_duration = generated_audio.shape[-1] / 24000
            rtf = processing_time / audio_duration
            
            # Save output
            audio_tensor = torch.from_numpy(generated_audio).unsqueeze(0)
            torchaudio.save(args.output, audio_tensor, 24000)
            
            print(f"RTF: {rtf:.3f}x ({1/rtf:.1f}x speed) | Processing: {processing_time:.2f}s | Audio: {audio_duration:.2f}s")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return 1
    


if __name__ == "__main__":
    sys.exit(main())