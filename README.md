# Jarod's Note

> Most likely will not be active on the repo for updates, etc if things break.

# DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/yl4579/DMOSpeech2)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.14988)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://dmospeech2.github.io/)

### Yinghao Aaron Li\*, Xilin Jiang\*, Fei Tao\*\*, Cheng Niu, Kaifeng Xu, Juntong Song, Nima Mesgarani

> Diffusion-based text-to-speech (TTS) systems have made remarkable progress in zero-shot speech synthesis, yet optimizing all components for perceptual metrics remains challenging. Prior work with DMOSpeech demonstrated direct metric optimization for speech generation components, but duration prediction remained unoptimized. This paper presents DMOSpeech 2, which extends metric optimization to the duration predictor through a reinforcement learning approach. The proposed system implements a novel duration policy framework using group relative preference optimization (GRPO) with speaker similarity and word error rate as reward signals. By optimizing this previously unoptimized component, DMOSpeech 2 creates a more complete metric-optimized synthesis pipeline. Additionally, this paper introduces teacher-guided sampling, a hybrid approach leveraging a teacher model for initial denoising steps before transitioning to the student model, significantly improving output diversity while maintaining efficiency. Comprehensive evaluations demonstrate superior performance across all metrics compared to previous systems, while reducing sampling steps by half without quality degradation. These advances represent a significant step toward speech synthesis systems with metric optimization across multiple components.
> *This work is accomplished in collaboration with Newsbreak.*

\*: Equal contribution
\*\*: Project leader

---

## TODO

* [ ] Fine-tune vocoder or train HiFTNet for higher acoustic quality

---

## Pre-requisites

* **uv (Astral)** installed – [https://docs.astral.sh/uv/#highlights](https://docs.astral.sh/uv/#highlights)
* **NVIDIA GPU** highly recommended

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/JarodMica/DMOSpeech2.git
   cd DMOSpeech2
   ```
2. Set up the environment with uv:

   ```bash
   uv sync
   ```

Models must be downloaded before inference.

---

## Inference

### 1. Get checkpoints with Git LFS (recommended)

**Windows:**

* Install Git LFS from [https://git-lfs.com/](https://git-lfs.com/)
* Then run:

```bash
git lfs install
git clone https://huggingface.co/yl4579/DMOSpeech ckpts
```

This pulls:

* `model_1500.pt` – GRPO-finetuned duration predictor
* `model_85000.pt` – DMOSpeech checkpoint (includes teacher for teacher-guided sampling)

> If `ckpts` already exists, either clone into a temp folder and move files, or rename/remove `ckpts` first.

### 2. (Alternative) Manual download

```bash
mkdir ckpts
cd ckpts
wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt
wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt
```

### 3. Run inference

```bash
uv run generate_audio.py
```

---

## TODO (Inference features)

* [ ] Streaming/Concatenating inference (like F5-TTS)

---

## Training

### Under construction

* [ ] Clean and test DMOSpeech training code
* [ ] Clean and test duration predictor pre-training
* [ ] Clean and test speaker verification and CTC model training
* [ ] Clean and test GRPO fine-tuning

---

## References

* [F5-TTS](https://github.com/SWivid/F5-TTS): Main codebase modified from F5-TTS repo, which also serves as the teacher
* [DMD2](https://github.com/tianweiy/DMD2): Training recipe
* [simple\_GRPO](https://github.com/lsdefine/simple_GRPO): GRPO training recipe
