# Shenava Live Persian Speech Recognition

Real-time Persian speech recognition using **Shenava Koochik FP32**, Sherpa-ONNX, and WebRTC VAD.

The application continuously listens to the microphone, detects speech segments, generates live text previews, and saves recognized Persian text to a file.

## Features

- Persian speech recognition using Shenava Koochik FP32.
- Sherpa-ONNX inference on CPU.
- Continuous microphone input.
- WebRTC Voice Activity Detection (VAD).
- Automatic background noise calibration.
- Live transcription preview.
- Persian right-to-left text rendering.
- Fixed terminal interface without scrolling.
- Continuous paragraph-style transcript display.
- Automatic transcript saving.
- Runtime noise calibration controls.

## Model

| Property | Value |
|---|---|
| Model | Shenava Koochik v1.0 |
| Model format | ONNX |
| Precision | FP32 |
| Architecture | FastConformer |
| Decoder | CTC |
| Inference engine | Sherpa-ONNX |
| Execution device | CPU |
| Sample rate | 16,000 Hz |
| Audio channels | Mono |

Model repository:

https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx

The model files are downloaded automatically on the first run.

## Requirements

- Python 3.12
- Linux or WSL2
- A working microphone
- Internet connection for the initial model download

Main dependencies:

- sherpa-onnx
- sounddevice
- webrtcvad-wheels
- numpy
- scipy
- huggingface-hub
- rich
- arabic-reshaper
- python-bidi

PyTorch and NVIDIA CUDA are not required for inference.

## Installation

Clone or download this repository.

Activate the Python virtual environment:

```bash
cd ~/AI/Shenava
source bin/activate
```

Enter the source directory:

```bash
cd main
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running

Start live speech recognition:

```bash
python live_shenava.py
```

To list available microphones:

```bash
python live_shenava.py --list-devices
```

To select a specific microphone:

```bash
python live_shenava.py --device 2
```

Replace `2` with the desired input device index.

## Terminal Interface

The application uses a fixed terminal interface.

Recognized text is displayed in a continuous paragraph instead of separate output panels.

The current utterance is shown as a live preview and replaced by its final recognized text when speech ends.

The terminal does not continuously scroll during normal operation.

## Keyboard Controls

| Key | Action |
|---|---|
| `C` | Recalibrate background noise |
| `D` | Disable noise calibration |
| `Q` | Quit the application |

### Noise Calibration

At startup, the application measures background noise for two seconds.

Remain silent during calibration.

The measured noise level is used to calculate the VAD energy threshold.

Press `C` whenever the environment or microphone noise level changes.

Press `D` to disable calibration and use the fixed energy threshold.

## Speech Recognition Pipeline

```text
Microphone
    |
    v
Audio Input
    |
    v
Sample Rate Conversion
    |
    v
WebRTC VAD
    |
    v
Speech Segmentation
    |
    v
Sherpa-ONNX
    |
    v
Shenava Koochik FP32
    |
    v
Persian Text
    |
    +------> Live Terminal
    |
    +------> transcriptions.txt
```

### Live Transcription

The recognition engine uses an offline CTC recognizer.

To provide a live preview, the application periodically reprocesses the accumulated audio of the current utterance.

The partial result may change as more audio becomes available.

After the end of the utterance is detected, the final result replaces the preview and is appended to the transcript.

This is incremental offline recognition, not true token-by-token streaming inference.

## Output Files

### transcriptions.txt

Contains the recognized Persian text as a continuous paragraph.

The original Unicode text is saved without terminal-specific RTL reshaping.

### Model files

Downloaded model files are stored in:

```text
models/
└── shenava-koochik-fp32/
    ├── model.onnx
    └── tokens.txt
```

## Configuration

The main settings are defined near the beginning of `live_shenava.py`.

| Parameter | Description |
|---|---|
| `VAD_MODE` | WebRTC VAD aggressiveness |
| `START_WINDOW_MS` | Speech detection window |
| `START_RATIO` | Required speech ratio to start an utterance |
| `PRE_ROLL_MS` | Audio retained before speech onset |
| `END_SILENCE_MS` | Silence duration required to finish an utterance |
| `NOISE_MULTIPLIER` | Background noise threshold multiplier |
| `MIN_FRAME_RMS` | Minimum audio energy threshold |
| `CPU_THREADS` | Number of CPU inference threads |
| `PARTIAL_INTERVAL` | Interval between live recognition updates |

## Notes

- The model expects mono 16 kHz audio.
- Microphone audio is resampled when necessary.
- Model inference runs on the CPU.
- The first run requires downloading the model files.
- Recognition speed depends on the CPU and audio duration.
- Live previews are provisional; the final result is used for the saved transcript.

## License

Check the licenses of the model, Sherpa-ONNX, and all third-party dependencies before redistributing this project or its model files.

---

**Shenava Live ASR**  
Persian Speech Recognition on CPU.