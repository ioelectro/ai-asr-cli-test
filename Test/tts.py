
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = Path("models/fa_IR-ganji-medium.onnx")


# --------------------------------------------------
# Audio device
# --------------------------------------------------

def select_stereo_output():
    """Select the default stereo output or find another stereo device."""

    try:
        info = sd.query_devices(kind="output")

        if info["max_output_channels"] >= 2:
            return None, info

    except Exception:
        pass

    devices = sd.query_devices()

    for index, device in enumerate(devices):
        if device["max_output_channels"] >= 2:
            return index, device

    raise RuntimeError(
        "No stereo output device found.\n"
        "Check your WSL audio configuration."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not MODEL.is_file():
        print(f"ERROR: Model not found: {MODEL}")
        return

    # Select audio output
    try:
        output_device, device_info = select_stereo_output()

        print("Audio output:")
        print(f"  Device   : {device_info['name']}")
        print(f"  Channels : 2")

    except Exception as e:
        print(f"Audio device error: {e}")
        return

    # Load Piper model
    print("\nLoading Persian Ganji model...")

    try:
        start = time.perf_counter()

        voice = PiperVoice.load(str(MODEL))

        load_time = time.perf_counter() - start

        print(f"Model loaded in {load_time:.2f}s")

    except Exception as e:
        print(f"Model loading error: {e}")
        return

    print("\nTTS Ready. Type 'exit' to quit.")

    try:

        while True:

            text = input("\nمتن فارسی: ").strip()

            if text.lower() in ("exit", "quit"):
                break

            if not text:
                continue

            try:

                # Generate speech
                start = time.perf_counter()

                chunks = list(voice.synthesize(text))

                if not chunks:
                    print("No audio generated.")
                    continue

                # Combine Piper chunks
                audio = np.concatenate([
                    chunk.audio_float_array
                    for chunk in chunks
                ]).astype(np.float32)

                sample_rate = chunks[0].sample_rate

                if audio.size == 0:
                    print("Empty audio.")
                    continue

                if not np.isfinite(audio).all():
                    print("Invalid audio samples.")
                    continue

                # Convert mono to dual-mono stereo
                stereo_audio = np.repeat(
                    audio[:, np.newaxis],
                    2,
                    axis=1
                )

                generation_time = time.perf_counter() - start
                duration = len(audio) / sample_rate

                print(
                    f"Generated : {generation_time:.3f}s\n"
                    f"Duration  : {duration:.2f}s\n"
                    f"SampleRate: {sample_rate} Hz\n"
                    f"Channels  : 2"
                )

                # Non-blocking playback
                # Do NOT pass channels=2.
                # sounddevice infers it from the array shape.
                sd.play(
                    stereo_audio,
                    samplerate=sample_rate,
                    device=output_device,
                    blocking=False
                )

                print("Playback started.")

            except Exception as e:
                print(f"TTS Error: {type(e).__name__}: {e}")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")

    finally:
        try:
            sd.stop()
        except Exception:
            pass

        print("TTS stopped.")


if __name__ == "__main__":
    main()

