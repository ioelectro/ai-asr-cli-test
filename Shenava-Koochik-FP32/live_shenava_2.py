#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import queue
import threading
import time
import sys
import select
import termios
import tty

from pathlib import Path
from collections import deque
from datetime import datetime
from math import gcd, ceil

import numpy as np
import sounddevice as sd
import webrtcvad
import sherpa_onnx

from scipy.signal import resample_poly
from huggingface_hub import snapshot_download

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich import box


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "shenava-koochik-fp32"
MODEL_FILE = MODEL_DIR / "model.onnx"
TOKENS_FILE = MODEL_DIR / "tokens.txt"
OUTPUT_FILE = BASE_DIR / "transcriptions.txt"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20

# Voice Activity Detection
VAD_MODE = 3

START_WINDOW_MS = 500
START_RATIO = 0.60

PRE_ROLL_MS = 300
END_SILENCE_MS = 800

# Noise calibration
CALIBRATION_SECONDS = 4

MIN_FRAME_RMS = 0.0002
NOISE_MULTIPLIER = 2.0

# Adaptive noise tracking
NOISE_ADAPT_RATE = 0.003
NOISE_MARGIN = 2.0

# Speech validation
MIN_VOICED_FRAMES = 8
MIN_UTTERANCE_MS = 400
MAX_UTTERANCE_SECONDS = 20

# Partial recognition
PARTIAL_INTERVAL = 1.5
PARTIAL_MIN_SECONDS = 1.5

# CPU
CPU_THREADS = 4

# Microphone
INPUT_DEVICE = None

AUDIO_QUEUE_SIZE = 500
VISIBLE_HISTORY = 10

console = Console(highlight=False)

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
asr_queue = queue.Queue()
result_queue = queue.Queue()

audio_dropped = [0]


# ============================================================
# Persian display
# ============================================================

def rtl(text):
    return text or ""


def make_rtl_text(text, style="white"):
    return Text(
        rtl(text),
        style=style,
        justify="right",
        overflow="fold",
    )


# ============================================================
# Terminal UI
# ============================================================

def make_layout(state, history):

    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=6),
        Layout(name="transcript", ratio=1, minimum_size=7),
        Layout(name="footer", size=3),
    )

    header = Text()
    header.append(" SHENAVA ", style="bold bright_cyan")
    header.append("LIVE TRANSCRIPTION", style="bold white")
    header.append("     KOOCHIK FP32", style="dim")

    layout["header"].update(
        Panel(
            Align.center(header),
            box=box.DOUBLE,
            border_style="cyan",
        )
    )

    table = Table.grid(padding=(0, 2))

    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_column(style="dim")
    table.add_column(style="bold cyan")

    status_style = (
        "bold green"
        if state["status"] == "Listening"
        else "bold yellow"
    )

    table.add_row(
        "STATUS",
        Text(state["status"], style=status_style),
        "MICROPHONE",
        state["device"],
    )

    table.add_row(
        "VAD",
        f"WebRTC / Mode {VAD_MODE}",
        "NOISE RMS",
        f"{state['noise_rms']:.5f}",
    )

    table.add_row(
        "ENERGY GATE",
        f"{state['threshold']:.5f}",
        "DROPPED",
        str(audio_dropped[0]),
    )

    layout["status"].update(
        Panel(
            table,
            title="[bold cyan]SYSTEM STATUS[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    items = []

    for timestamp, text in history:
        items.append(
            Text(f"  {timestamp}", style="dim")
        )

        items.append(
            Align.right(make_rtl_text(text))
        )

    if state["speaking"]:

        items.append(
            Text("  LIVE PREVIEW", style="bold yellow")
        )

        if state["partial_text"]:
            items.append(
                Align.right(
                    make_rtl_text(
                        state["partial_text"],
                        "bold bright_white",
                    )
                )
            )
        else:
            items.append(
                Align.right(
                    make_rtl_text(
                        "در حال شنیدن گفتار...",
                        "dim",
                    )
                )
            )

    elif not history:

        items.append(
            Align.right(
                make_rtl_text(
                    "هنوز گفتاری ثبت نشده است.",
                    "dim",
                )
            )
        )

    layout["transcript"].update(
        Panel(
            Group(*items),
            title="[bold green]TRANSCRIPT[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    footer = Text(
        " C  Recalibrate     D  Disable calibration     Q  Quit",
        style="bold white",
        justify="center",
    )

    layout["footer"].update(
        Panel(
            footer,
            border_style="blue",
            box=box.ROUNDED,
        )
    )

    return layout


# ============================================================
# Model
# ============================================================

def load_model():

    if not MODEL_FILE.exists() or not TOKENS_FILE.exists():

        console.print("[yellow]Downloading Koochik FP32...[/yellow]")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=[
                "model.onnx",
                "tokens.txt",
                "persian_itn.py",
            ],
        )

    console.print("[cyan]Initializing Koochik FP32...[/cyan]")

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(MODEL_FILE),
        tokens=str(TOKENS_FILE),
        num_threads=CPU_THREADS,
        decoding_method="greedy_search",
        debug=False,
    )

    console.print("[bold green]Model loaded successfully.[/bold green]")

    return recognizer


# ============================================================
# Audio utilities
# ============================================================

def resample_audio(audio, input_rate):

    audio = np.asarray(audio, dtype=np.float32)

    if input_rate == SAMPLE_RATE:
        return audio

    divisor = gcd(input_rate, SAMPLE_RATE)

    return np.asarray(
        resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            input_rate // divisor,
        ),
        dtype=np.float32,
    )


def prepare_frame(frame, input_rate):

    audio = resample_audio(frame, input_rate)

    expected = SAMPLE_RATE * FRAME_MS // 1000

    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))

    elif len(audio) > expected:
        audio = audio[:expected]

    return np.clip(audio, -1.0, 1.0)


def calculate_rms(audio):

    return float(np.sqrt(np.mean(audio ** 2)))


def audio_callback(indata, frames, time_info, status):

    try:
        audio_queue.put_nowait(indata[:, 0].copy())

    except queue.Full:

        audio_dropped[0] += 1

        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Empty:
            pass


def clear_audio_queue():

    while True:

        try:
            audio_queue.get_nowait()

        except queue.Empty:
            break


# ============================================================
# Noise calibration
# ============================================================

def calibrate_noise(input_rate, progress_callback=None):

    levels = []

    target_frames = int(
        CALIBRATION_SECONDS * 1000 / FRAME_MS
    )

    deadline = time.monotonic() + CALIBRATION_SECONDS + 10

    while len(levels) < target_frames:

        if time.monotonic() > deadline:
            raise RuntimeError("Noise calibration timed out.")

        try:
            block = audio_queue.get(timeout=0.1)

        except queue.Empty:
            continue

        audio = prepare_frame(block, input_rate)

        rms = calculate_rms(audio)

        levels.append(rms)

        if progress_callback:
            progress_callback(len(levels) / target_frames)

    # صدک 75 برای تخمین محافظه‌کارانه‌تر نویز محیط
    noise_rms = float(np.percentile(levels, 75))

    threshold = max(
        MIN_FRAME_RMS,
        noise_rms * NOISE_MULTIPLIER,
    )

    return noise_rms, threshold


# ============================================================
# Voice Segmenter
# ============================================================

class VoiceSegmenter:

    def __init__(self, input_rate, noise_rms):

        self.input_rate = input_rate

        self.noise_rms = noise_rms

        self.energy_threshold = max(
            MIN_FRAME_RMS,
            noise_rms * NOISE_MULTIPLIER,
        )

        self.vad = webrtcvad.Vad(VAD_MODE)

        self.pre_roll = deque(
            maxlen=PRE_ROLL_MS // FRAME_MS
        )

        self.start_window = deque(
            maxlen=START_WINDOW_MS // FRAME_MS
        )

        self.frames = []

        self.voiced_frames = 0
        self.silence_frames = 0

        self.active = False

        self.sequence = 0
        self.current_id = None

        self.last_partial_time = 0.0

    # --------------------------------------------------------

    def reset(self):

        self.pre_roll.clear()
        self.start_window.clear()
        self.frames.clear()

        self.voiced_frames = 0
        self.silence_frames = 0

        self.active = False
        self.current_id = None

        self.last_partial_time = 0.0

    # --------------------------------------------------------

    def update_noise(self, rms):

        # نویز فقط زمانی به‌روزرسانی می‌شود
        # که قطعه گفتاری فعالی وجود ندارد.

        self.noise_rms = (
            (1.0 - NOISE_ADAPT_RATE) * self.noise_rms
            + NOISE_ADAPT_RATE * rms
        )

        self.energy_threshold = max(
            MIN_FRAME_RMS,
            self.noise_rms * NOISE_MARGIN,
        )

    # --------------------------------------------------------

    def detect_speech(self, frame):

        audio = prepare_frame(frame, self.input_rate)

        rms = calculate_rms(audio)

        # گیت انرژی
        if rms < self.energy_threshold:
            return False, rms

        pcm = (audio * 32767).astype("<i2").tobytes()

        try:
            vad_result = self.vad.is_speech(
                pcm,
                SAMPLE_RATE,
            )

        except Exception:
            vad_result = False

        return vad_result, rms

    # --------------------------------------------------------

    def finish(self):

        if not self.active or not self.frames:
            self.reset()
            return None

        uid = self.current_id

        duration_ms = len(self.frames) * FRAME_MS

        audio = np.concatenate(
            self.frames
        ).astype(np.float32)

        valid = (
            duration_ms >= MIN_UTTERANCE_MS
            and self.voiced_frames >= MIN_VOICED_FRAMES
        )

        self.reset()

        if not valid:
            return None

        return uid, audio

    # --------------------------------------------------------

    def push(self, frame):

        speech, rms = self.detect_speech(frame)

        # --------------------------------------------
        # Waiting for speech
        # --------------------------------------------

        if not self.active:

            # به‌روزرسانی تدریجی نویز محیط
            if not speech:
                self.update_noise(rms)

            self.pre_roll.append(
                (frame.copy(), speech)
            )

            self.start_window.append(speech)

            required = ceil(
                len(self.start_window) * START_RATIO
            )

            if (
                len(self.start_window) == self.start_window.maxlen
                and sum(self.start_window) >= required
            ):

                self.sequence += 1

                self.current_id = self.sequence
                self.active = True

                self.frames = [
                    item[0] for item in self.pre_roll
                ]

                self.voiced_frames = sum(
                    item[1] for item in self.pre_roll
                )

                self.silence_frames = 0

                self.last_partial_time = time.monotonic()

                return "started", self.current_id, None

            return None, None, None

        # --------------------------------------------
        # Active speech
        # --------------------------------------------

        self.frames.append(frame.copy())

        if speech:

            self.voiced_frames += 1
            self.silence_frames = 0

        else:

            self.silence_frames += 1

        duration = (
            len(self.frames) * FRAME_MS / 1000
        )

        silence_ms = (
            self.silence_frames * FRAME_MS
        )

        # --------------------------------------------
        # End of sentence
        # --------------------------------------------

        if (
            silence_ms >= END_SILENCE_MS
            or duration >= MAX_UTTERANCE_SECONDS
        ):

            final = self.finish()

            if final is None:
                return "discarded", None, None

            uid, audio = final

            return "final", uid, audio

        # --------------------------------------------
        # Partial recognition
        # --------------------------------------------

        now = time.monotonic()

        if (
            duration >= PARTIAL_MIN_SECONDS
            and self.voiced_frames >= MIN_VOICED_FRAMES
            and now - self.last_partial_time >= PARTIAL_INTERVAL
        ):

            self.last_partial_time = now

            partial = np.concatenate(
                self.frames
            ).astype(np.float32)

            return "partial", self.current_id, partial

        return None, None, None


# ============================================================
# ASR Worker
# ============================================================

def asr_worker(recognizer, input_rate):

    while True:

        task = asr_queue.get()

        if task is None:
            break

        kind, uid, audio = task

        try:

            audio_16k = resample_audio(
                audio,
                input_rate,
            )

            audio_16k = np.clip(
                audio_16k,
                -1.0,
                1.0,
            )

            stream = recognizer.create_stream()

            stream.accept_waveform(
                SAMPLE_RATE,
                audio_16k,
            )

            started = time.perf_counter()

            recognizer.decode_stream(stream)

            elapsed = time.perf_counter() - started

            text = stream.result.text.strip()

            duration = len(audio_16k) / SAMPLE_RATE

            result_queue.put(
                (kind, uid, text, duration, elapsed)
            )

        except Exception as error:

            result_queue.put(
                ("error", uid, str(error), 0, 0)
            )


# ============================================================
# Result handling
# ============================================================

def process_results(state, history):

    while True:

        try:
            result = result_queue.get_nowait()

        except queue.Empty:
            break

        kind, uid, text, duration, elapsed = result

        if kind == "error":

            state["status"] = "Recognition error"
            state["message"] = text

            continue

        if kind == "partial":

            if uid == state["active_id"] and text:

                state["partial_text"] = text
                state["status"] = "Live transcription"

            continue

        if kind == "final":

            if text:

                timestamp = datetime.now().strftime("%H:%M:%S")

                with OUTPUT_FILE.open(
                    "a",
                    encoding="utf-8",
                ) as file:

                    file.write(
                        f"[{timestamp}] {text}\n"
                    )

                    file.flush()

                history.append(
                    (timestamp, text)
                )

                state["last_rtf"] = (
                    elapsed / duration if duration else 0
                )

                state["last_inference"] = elapsed

            if state["active_id"] == uid:

                state["active_id"] = None
                state["partial_text"] = ""
                state["speaking"] = False
                state["status"] = "Listening"

        elif kind == "discarded":

            if state["active_id"] == uid:

                state["active_id"] = None
                state["partial_text"] = ""
                state["speaking"] = False
                state["status"] = "Listening"


# ============================================================
# Keyboard
# ============================================================

def read_key():

    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select(
        [sys.stdin],
        [],
        [],
        0,
    )

    if ready:
        return sys.stdin.read(1).lower()

    return None


# ============================================================
# Microphones
# ============================================================

def list_microphones():

    console.print(
        "\n[bold cyan]Available microphones[/bold cyan]\n"
    )

    for index, device in enumerate(sd.query_devices()):

        if device["max_input_channels"] > 0:

            console.print(
                f"{index}: {device['name']} "
                f"({device['default_samplerate']} Hz)"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=INPUT_DEVICE,
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
        return

    recognizer = load_model()

    device = sd.query_devices(
        args.device,
        "input",
    )

    input_rate = int(
        round(device["default_samplerate"])
    )

    frame_samples = round(
        input_rate * FRAME_MS / 1000
    )

    history = deque(maxlen=VISIBLE_HISTORY)

    state = {
        "status": "Starting",
        "device": device["name"],
        "threshold": MIN_FRAME_RMS,
        "noise_rms": 0.0,
        "calibrated": False,
        "speaking": False,
        "partial_text": "",
        "active_id": None,
        "message": "",
        "last_rtf": 0.0,
        "last_inference": 0.0,
    }

    segmenter = VoiceSegmenter(
        input_rate,
        0.0,
    )

    worker = threading.Thread(
        target=asr_worker,
        args=(recognizer, input_rate),
        daemon=True,
    )

    worker.start()

    old_terminal = None

    if sys.stdin.isatty():

        fd = sys.stdin.fileno()

        old_terminal = termios.tcgetattr(fd)

        tty.setcbreak(fd)

    try:

        initial_layout = make_layout(
            state,
            history,
        )

        with Live(
            initial_layout,
            console=console,
            screen=True,
            refresh_per_second=10,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:

            try:

                with sd.InputStream(
                    device=args.device,
                    samplerate=input_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=frame_samples,
                    callback=audio_callback,
                ):

                    # ------------------------------------
                    # Initial calibration
                    # ------------------------------------

                    state["status"] = (
                        "Calibrating — stay silent"
                    )

                    clear_audio_queue()

                    def calibration_progress(progress):

                        state["status"] = (
                            f"Calibrating {progress * 100:.0f}% "
                            "- stay silent"
                        )

                        live.update(
                            make_layout(state, history),
                            refresh=True,
                        )

                    noise_rms, threshold = calibrate_noise(
                        input_rate,
                        calibration_progress,
                    )

                    segmenter.noise_rms = noise_rms

                    segmenter.energy_threshold = threshold

                    state["noise_rms"] = noise_rms
                    state["threshold"] = threshold
                    state["calibrated"] = True
                    state["status"] = "Listening"

                    clear_audio_queue()

                    # ------------------------------------
                    # Main loop
                    # ------------------------------------

                    while True:

                        process_results(
                            state,
                            history,
                        )

                        key = read_key()

                        # Recalibrate
                        if key == "c":

                            segmenter.reset()

                            state["active_id"] = None
                            state["speaking"] = False
                            state["partial_text"] = ""

                            state["status"] = (
                                "Calibrating — stay silent"
                            )

                            clear_audio_queue()

                            live.update(
                                make_layout(state, history),
                                refresh=True,
                            )

                            noise_rms, threshold = calibrate_noise(
                                input_rate,
                                calibration_progress,
                            )

                            segmenter.noise_rms = noise_rms
                            segmenter.energy_threshold = threshold

                            state["noise_rms"] = noise_rms
                            state["threshold"] = threshold
                            state["calibrated"] = True
                            state["status"] = "Listening"

                            clear_audio_queue()

                            continue

                        # Disable calibration
                        elif key == "d":

                            segmenter.reset()

                            segmenter.noise_rms = 0.0
                            segmenter.energy_threshold = MIN_FRAME_RMS

                            state["noise_rms"] = 0.0
                            state["threshold"] = MIN_FRAME_RMS

                            state["calibrated"] = False
                            state["speaking"] = False
                            state["active_id"] = None
                            state["partial_text"] = ""
                            state["status"] = "Listening"

                            clear_audio_queue()

                            continue

                        # Quit
                        elif key == "q":
                            break

                        try:

                            block = audio_queue.get(
                                timeout=0.05
                            )

                        except queue.Empty:

                            live.update(
                                make_layout(state, history)
                            )

                            continue

                        event, uid, audio = segmenter.push(
                            block
                        )

                        # Speech started
                        if event == "started":

                            state["active_id"] = uid
                            state["speaking"] = True
                            state["partial_text"] = ""
                            state["status"] = "Speech detected"

                        # Partial
                        elif event == "partial":

                            state["speaking"] = True

                            if asr_queue.qsize() < 3:

                                asr_queue.put_nowait(
                                    ("partial", uid, audio)
                                )

                        # Final
                        elif event == "final":

                            state["status"] = "Finalizing"
                            state["speaking"] = False

                            asr_queue.put(
                                ("final", uid, audio)
                            )

                        # Discarded segment
                        elif event == "discarded":

                            state["active_id"] = None
                            state["speaking"] = False
                            state["partial_text"] = ""
                            state["status"] = "Listening"

                        # Show current adaptive threshold
                        state["noise_rms"] = segmenter.noise_rms
                        state["threshold"] = segmenter.energy_threshold

                        live.update(
                            make_layout(state, history)
                        )

            except KeyboardInterrupt:

                state["status"] = "Stopping"

            finally:

                remaining = segmenter.finish()

                if remaining is not None:

                    uid, audio = remaining

                    asr_queue.put(
                        ("final", uid, audio)
                    )

                asr_queue.put(None)

                worker.join()

                process_results(
                    state,
                    history,
                )

                state["status"] = "Stopped"
                state["speaking"] = False
                state["partial_text"] = ""

                live.update(
                    make_layout(state, history),
                    refresh=True,
                )

    finally:

        if old_terminal is not None:

            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                old_terminal,
            )

        console.print(
            f"\n[green]Transcript saved:[/green] "
            f"{OUTPUT_FILE.resolve()}"
        )


if __name__ == "__main__":
    main()