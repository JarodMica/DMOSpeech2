# Jarod's Note:
> Most likely will not be active on the repo for updates, etc if things break.

# DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/yl4579/DMOSpeech2)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.14988)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://dmospeech2.github.io/)

### Yinghao Aaron Li*, Xilin Jiang*, Fei Tao**, Cheng Niu, Kaifeng Xu, Juntong Song, Nima Mesgarani

> Diffusion-based text-to-speech (TTS) systems have made remarkable progress in zero-shot speech synthesis, yet optimizing all components for perceptual metrics remains challenging. Prior work with DMOSpeech demonstrated direct metric optimization for speech generation components, but duration prediction remained unoptimized. This paper presents DMOSpeech 2, which extends metric optimization to the duration predictor through a reinforcement learning approach. The proposed system implements a novel duration policy framework using group relative preference optimization (GRPO) with speaker similarity and word error rate as reward signals. By optimizing this previously unoptimized component, DMOSpeech 2 creates a more complete metric-optimized synthesis pipeline. Additionally, this paper introduces teacher-guided sampling, a hybrid approach leveraging a teacher model for initial denoising steps before transitioning to the student model, significantly improving output diversity while maintaining efficiency. Comprehensive evaluations demonstrate superior performance across all metrics compared to previous systems, while reducing sampling steps by half without quality degradation. These advances represent a significant step toward speech synthesis systems with metric optimization across multiple components.
>
> _This work is accomplished in collaborating with Newsbreak._

*: Equal contribution

**: Project leader

TODO: 

- [ ] Fine-tune vocoder or train HiFTNet for higher acoustic quality

## Pre-requisites
- uv astral installed on your computer - https://docs.astral.sh/uv/#highlights
- An NVIDIA Graphics card highly recommended

### Installation

1. Clone the repo:
  ```
  git clone https://github.com/JarodMica/DMOSpeech2.git
  cd DMOSpeech2
  ```
2. Set-up env. with uv
  ```
  uv sync
  ```

This should pretty much be it, models will need to be downloaded before inference.

## Inference

1. Download checkpoints from [Huggingface](https://huggingface.co/yl4579/DMOSpeech2) to `ckpts` folder. Create a new `ckpts` folder in the root if there isn't one.
  - [model_1500.pt](https://huggingface.co/yl4579/DMOSpeech2/blob/main/model_1500.pt) is the GRPO-finetuned duration predictor checkpoint.
  - [model_85000.pt](https://huggingface.co/yl4579/DMOSpeech2/blob/main/model_85000.pt) is the DMOSpeech checkpoint (including teacher for teacher-guided sampling).

You can also run the following command lines:

  ```bash
  mkdir ckpts
  cd ckpts
  wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt
  wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt
  ```

2. Test out inference with:
  ```
  uv run generate_audio.py
  ```

TODO: 

- [ ] Streaming/Concatenating inference (like F5-TTS)

## Training

### Under construction

TODO: 

- [ ] Clean and test DMOSpeech training code
- [ ] Clean and test the duration predictor pre-training
- [ ] Clean and test speaker verification and CTC model training
- [ ] Clean and test GRPO fine-tuning

## References

- [F5-TTS](https://github.com/SWivid/F5-TTS): Main codebase modified from F5-TTS repo, which also serves as the teacher
- [DMD2](https://github.com/tianweiy/DMD2): Training recipe
- [simple_GRPO](https://github.com/lsdefine/simple_GRPO): GRPO training recipe

