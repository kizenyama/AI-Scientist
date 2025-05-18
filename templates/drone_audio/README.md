# Drone Audio Detection Using AI Scientist

This repository works within AI Scientist framework, implementing TinyML models for drone audio detection, optimizing CNN architectures for embedded devices.

## AI Scientist Generated Artefacts

While the full paper pdf failed to save due to latex error on my local environment, the system completed the idea generation, experimentation and write up phase, so I share the artefacts.
[Link to Artefacts](https://drive.google.com/drive/folders/1-3mKdXB5a1uLbuP23slwd19tBPxFXwzc?usp=drive_link)

## Installation

The full list of installation requirements is at the first segment of experiment.py.

## Dataset Preparation

The dataset includes drone propeller noises recorded indoors by Sara Al Emadi (2019). The dataset preparation code is included in the experiment.py


## Running AI Scientist

1. Initialize and run the AI Scientist:

```bash
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment drone_audio \
    --num-ideas 1
```

## Credits

This project builds upon the following works:

- **The AI Scientist System**
    - Repository: https://github.com/SakanaAI/AI-Scientist

- **Drone Audio Dataset Source**
    - Repository: https://github.com/saraalemadi/DroneAudioDataset
