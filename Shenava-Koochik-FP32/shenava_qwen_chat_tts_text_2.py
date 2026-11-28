#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shenava Persian voice + text chat (Sherpa-ONNX, Ollama, Piper)."""

import argparse
import json
import os
import locale
import queue
import re
import select
import socket
import sys
import termios
import threading
import time
import tty
import urllib.request
from urllib.parse import urlsplit
from collections import deque
from datetime import datetime
from math import ceil, gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import sherpa_onnx
from scipy.signal import resample_poly, butter, sosfiltfilt
from huggingface_hub import snapshot_download

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich import box

# wcwidth is optional; this fallback keeps Shenava runnable without extra packages.
try:
    from wcwidth import wcswidth
except ImportError:
    import unicodedata

    def wcswidth(value):
        width = 0
        for char in str(value):
            category = unicodedata.category(char)
            if category.startswith("C") or unicodedata.combining(char):
                continue
            width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        return width


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_TIMEOUT = 300
OLLAMA_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SYSTEM_PROMPT = (
    "طبیعی، دقیق و کوتاه پاسخ بده."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "shenava-koochik-fp32"
MODEL_FILE = MODEL_DIR / "model.onnx"
TOKENS_FILE = MODEL_DIR / "tokens.txt"
OUTPUT_FILE = BASE_DIR / "transcriptions.txt"

TTS_MODEL_FILE = BASE_DIR / "models" / "fa_IR-ganji-medium.onnx"
TTS_OUTPUT_DEVICE = None
TTS_ENABLED_BY_DEFAULT = True
TTS_QUEUE_SIZE = 0  # 0 = unbounded queue for streamed speech
TTS_STREAM_WORDS = 3  # Speak after 3 complete words, or earlier at punctuation

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20

# Aggressive mode reduces false speech detections from steady room noise.
VAD_MODE = 2                    # Balanced: less likely to reject quiet speech.
START_WINDOW_MS = 400
START_RATIO = 0.45
PRE_ROLL_MS = 360
END_SILENCE_MS = 400

CALIBRATION_SECONDS = 4
MIN_FRAME_RMS = 0.0005
NOISE_MULTIPLIER = 1.30
NOISE_UPDATE_ALPHA = 0.02
END_NOISE_MULTIPLIER = 1.70
CONTINUE_NOISE_MULTIPLIER = 1.12
END_SPEECH_RATIO = 0.32
MAX_ENERGY_THRESHOLD = 0.04

MIN_VOICED_FRAMES = 7
MIN_UTTERANCE_MS = 450
MAX_UTTERANCE_SECONDS = 20

PARTIAL_INTERVAL = 1.0
PARTIAL_MIN_SECONDS = 1.0
CPU_THREADS = 4
INPUT_DEVICE = None

AUDIO_QUEUE_SIZE = 500
VISIBLE_HISTORY = 50

console = Console(highlight=False, soft_wrap=True)

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
asr_queue = queue.Queue()
result_queue = queue.Queue()
chat_queue = queue.Queue()
chat_result_queue = queue.Queue()
tts_queue = queue.Queue(maxsize=TTS_QUEUE_SIZE)
TTS_STREAMING_ENABLED = False
tts_result_queue = queue.Queue()
cancel_event = threading.Event()
tts_playing = threading.Event()
audio_dropped = [0]
audio_status = [""]


# ============================================================
# Persian display helpers
# Keep Unicode unchanged; directional marks only set the paragraph direction.
# ============================================================

def rtl(value):
    """Keep original letters and tell the terminal this line is RTL."""
    return "\u200f" + str(value) if value is not None else ""


def ltr(value):
    """Keep timestamps and numeric labels left-to-right."""
    return "\u200e" + str(value)


def model_display_name():
    """Return the actual Ollama model ID for the English UI."""
    return OLLAMA_MODEL


def cell_width(value):
    """Return terminal-cell width, not Python character count."""
    width = wcswidth(value)
    return width if width >= 0 else len(value)


def wrap_logical(value, width):
    """Wrap text without modifying its Unicode character order."""
    words = str(value).split()
    if not words:
        return [""]

    lines = []
    current = ""
    for word in words:
        # Split an unusually long token without exceeding terminal width.
        pieces = []
        piece = ""
        for char in word:
            if piece and cell_width(piece + char) > width:
                pieces.append(piece)
                piece = char
            else:
                piece += char
        if piece:
            pieces.append(piece)

        for part in pieces:
            candidate = f"{current} {part}" if current else part
            if current and cell_width(candidate) > width:
                lines.append(current)
                current = part
            else:
                current = candidate
    if current or not lines:
        lines.append(current)
    return lines


def rtl_renderable(value, width=None, style="white"):
    if width is None:
        width = max(10, console.width - 8)

    lines = []
    for paragraph in str(value).splitlines() or [""]:
        lines.extend(rtl(line) for line in wrap_logical(paragraph, width))

    return Group(*[
        Align.right(Text(line, style=style, no_wrap=True, overflow="crop"), pad=True)
        for line in lines
    ])


def append_rtl_lines(target, value, width, style="white"):
    """Preserve original Unicode order; do not reshape or reverse Persian text."""
    # Rich wraps the original text; the terminal performs RTL/BiDi rendering.
    target.append((rtl(value), style, True))


# ============================================================
# Terminal interface
# ============================================================

def make_layout(state, history):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=8),
        Layout(name="transcript", ratio=1, minimum_size=7),
        Layout(name="input", size=3),
        Layout(name="footer", size=3),
    )

    header = Text()
    header.append(" SHENAVA ", style="bold bright_cyan")
    header.append("VOICE + TEXT CHAT", style="bold white")
    header.append(f"     KOOCHIK FP32 | {model_display_name()}", style="dim")
    layout["header"].update(
        Panel(Align.center(header), box=box.DOUBLE,
              border_style="cyan", padding=(0, 1))
    )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_column(style="dim")
    table.add_column(style="bold cyan")

    status_style = "bold green" if state["status"] == "Listening" else "bold yellow"
    table.add_row("STATUS", Text(state["status"], style=status_style),
                  "MICROPHONE", state["device"])
    calibration = (
        f"ACTIVE ({state['threshold']:.4f})"
        if state["calibrated"] else f"DISABLED ({state['threshold']:.4f})"
    )
    table.add_row("VAD", f"WebRTC mode {VAD_MODE}", "CALIBRATION", calibration)
    table.add_row(
        "NOISE", f"{state.get('noise_floor', 0.0):.4f} RMS | Drop {audio_dropped[0]}",
        "ENDPOINT", f"{state.get('silence_ms', 0)} / {END_SILENCE_MS} ms",
    )

    ai_style = "bold cyan" if state["ai_busy"] else "dim"
    table.add_row("AI", Text(state["ai_status"], style=ai_style),
                  "MODEL", model_display_name())

    tts_state = state["tts_status"]
    tts_style = (
        "bold green" if tts_state == "Speaking" or tts_state.startswith("Ready")
        else ("dim" if tts_state == "Disabled" else "bold yellow")
    )
    table.add_row("TTS", Text(tts_state, style=tts_style),
                  "VOICE", "Ganji Medium")
    layout["status"].update(
        Panel(table, title="[bold cyan]SYSTEM STATUS[/bold cyan]",
              border_style="cyan", box=box.ROUNDED)
    )

    # Keep Persian text untouched so the terminal handles BiDi rendering.
    width = max(12, console.width - 10)
    transcript = []

    for timestamp, message in history:
        transcript.append((ltr(f"  {timestamp}"), "dim", False))
        # Keep the English Qwen role marker at the far right of its chat entry.
        if message.startswith("Qwen: "):
            transcript.append((ltr("Qwen:"), "bold bright_cyan", "right_ltr"))
            append_rtl_lines(transcript, message[6:], width, "white")
        else:
            append_rtl_lines(transcript, message, width, "white")

    if state["speaking"]:
        transcript.append((ltr("LIVE PREVIEW"), "bold yellow", False))
        preview = state["partial_text"] or "در حال شنیدن گفتار..."
        append_rtl_lines(transcript, preview, width, "bold bright_white")

    if state["ai_busy"]:
        transcript.append((ltr("QWEN STREAMING"), "bold bright_cyan", "right_ltr"))
        stream = state["stream_text"] or "در انتظار پاسخ..."
        append_rtl_lines(transcript, stream, width, "bright_white")

    if not transcript:
        transcript.append((rtl("هنوز پیامی ثبت نشده است."), "dim", True))

    viewport = max(1, console.height - 21)
    max_offset = max(0, len(transcript) - viewport)
    offset = min(max(0, state["scroll_offset"]), max_offset)
    state["scroll_offset"] = offset
    end = len(transcript) - offset
    visible = transcript[max(0, end - viewport):end]

    rendered = []
    for line, style, is_rtl in visible:
        item = Text(
            line, style=style,
            justify="right" if is_rtl else "left",
            no_wrap=False, overflow="fold",
        )
        # Timestamps stay on the left; Persian messages stay on the right.
        if is_rtl == "right_ltr":
            rendered.append(Align.right(item, pad=True))
        else:
            rendered.append(
                Align.right(item, pad=True) if is_rtl else Align.left(item, pad=True)
            )

    scroll_label = "LIVE" if offset == 0 else f"SCROLL: {offset}"
    layout["transcript"].update(
        Panel(
            Group(*rendered),
            title="[bold green]TRANSCRIPT[/bold green]",
            subtitle=f"[dim]{scroll_label} | ↑/↓ scroll | PgUp/PgDn page | End latest[/dim]",
            border_style="green", box=box.ROUNDED, padding=(1, 2),
        )
    )

    # The input is a single-line editor; characters are appended at the end.
    input_value = state["input_text"]
    if input_value:
        cursor = max(0, min(state.get("input_cursor", len(input_value)), len(input_value)))
        prompt = input_value[:cursor] + "▏" + input_value[cursor:]
    else:
        prompt = "Type here..."
    layout["input"].update(
        Panel(
            rtl_renderable(prompt + "  ", width=max(10, console.width - 10)),
            title="[bold cyan]TEXT INPUT[/bold cyan]",
            border_style="cyan", padding=(0, 1),
        )
    )

    footer = Text(
        f"Silence {END_SILENCE_MS}ms | ENTER Send | BACKSPACE Delete | CTRL+U Clear | ESC Cancel | "
        "↑/↓ Scroll | ←/→ Cursor | Home/End | C Calibrate | D Disable | Q Quit",
        style="bold white", justify="center",
    )
    layout["footer"].update(
        Panel(footer, border_style="blue", box=box.ROUNDED,
              padding=(0, 1))
    )
    return layout


# ============================================================
# Model loading
# ============================================================

def load_model():
    if not MODEL_FILE.exists() or not TOKENS_FILE.exists():
        console.print("[yellow]Downloading Koochik FP32...[/yellow]")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        proxy_names = [
            name for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
            if os.environ.get(name)
        ]
        if proxy_names:
            console.print(
                "[dim]Hugging Face download will use proxy environment: "
                + ", ".join(proxy_names) + "[/dim]"
            )
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=str(MODEL_DIR),
                allow_patterns=["model.onnx", "tokens.txt", "persian_itn.py"],
            )
        except Exception as error:
            raise RuntimeError(
                "Koochik download failed. Check HTTPS_PROXY/HTTP_PROXY and "
                "confirm that the proxy can reach huggingface.co. "
                f"Details: {type(error).__name__}: {error}"
            ) from error

    console.print("[cyan]Initializing Koochik FP32...[/cyan]")
    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(MODEL_FILE),
        tokens=str(TOKENS_FILE),
        num_threads=CPU_THREADS,
        decoding_method="greedy_search",
        debug=False,
    )
    console.print("[bold green]ASR model loaded.[/bold green]")
    return recognizer


# ============================================================
# Audio helpers
# ============================================================

def resample_audio(audio, input_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if input_rate == SAMPLE_RATE:
        return audio

    divisor = gcd(int(input_rate), SAMPLE_RATE)
    return np.asarray(
        resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(input_rate) // divisor,
        ),
        dtype=np.float32,
    )


# Speech band-pass: remove handling noise / low-frequency rumble before ASR.
_ASR_FILTER = butter(3, [85, 7600], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def prepare_asr_audio(audio):
    """Clean DC and out-of-band noise while preserving natural speech levels."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = np.nan_to_num(audio, copy=False)
    audio = audio - float(np.mean(audio))
    if audio.size > 96:
        try:
            audio = sosfiltfilt(_ASR_FILTER, audio).astype(np.float32, copy=False)
        except ValueError:
            pass
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def audio_callback(indata, frames, time_info, status):
    # Ignore microphone input while Piper is speaking.
    if status:
        audio_status[0] = str(status)
    if tts_playing.is_set():
        return

    block = indata[:, 0].copy()
    try:
        audio_queue.put_nowait(block)
    except queue.Full:
        audio_dropped[0] += 1
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            audio_dropped[0] += 1


def clear_audio_queue():
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# ============================================================
# Piper text-to-speech
# ============================================================

def load_tts_model():
    config_file = Path(str(TTS_MODEL_FILE) + ".json")
    if not TTS_MODEL_FILE.is_file() or not config_file.is_file():
        console.print("[yellow]Piper model/config not found; TTS disabled.[/yellow]")
        return None

    try:
        from piper import PiperVoice
        console.print("[cyan]Loading Persian Ganji TTS...[/cyan]")
        started = time.perf_counter()
        voice = PiperVoice.load(str(TTS_MODEL_FILE))
        console.print(f"[green]TTS loaded in {time.perf_counter() - started:.2f}s[/green]")
        return voice
    except Exception as error:
        console.print(f"[yellow]TTS disabled: {error}[/yellow]")
        return None


def clean_text_for_tts(value):
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"(?m)^\s*[-*+]\s+", "", value)
    value = re.sub(r"[*_#>`~]", "", value)
    return re.sub(r"\s+", " ", value).strip()


class StreamingSpeechBuffer:
    """Turn incoming model tokens into short, pronounceable TTS phrases."""

    def __init__(self, words_per_chunk=3):
        self.words_per_chunk = max(2, int(words_per_chunk))
        self.buffer = ""

    def feed(self, piece="", final=False):
        self.buffer += str(piece or "")
        ready = []

        while self.buffer.strip():
            # Prefer natural pauses when punctuation arrives.
            punctuation = re.search(r"[.!?؟؛،:\n]", self.buffer)
            if punctuation:
                end = punctuation.end()
                phrase = self.buffer[:end]
                word_count = len(re.findall(r"\S+", phrase))
                strong_pause = punctuation.group(0) in ".!?؟؛;:\n"
                if word_count >= 2 or strong_pause:
                    cleaned = clean_text_for_tts(phrase)
                    if cleaned:
                        ready.append(cleaned)
                    self.buffer = self.buffer[end:]
                    continue

            # Without punctuation, release a few complete words at a time.
            words = list(re.finditer(r"\S+", self.buffer))
            count = self.words_per_chunk
            if len(words) >= count:
                end = words[count - 1].end()
                # Wait for whitespace after the last word so tokens are not cut.
                if end < len(self.buffer) and self.buffer[end].isspace():
                    cleaned = clean_text_for_tts(self.buffer[:end])
                    if cleaned:
                        ready.append(cleaned)
                    self.buffer = self.buffer[end:]
                    continue

            break

        if final and self.buffer.strip():
            cleaned = clean_text_for_tts(self.buffer)
            if cleaned:
                ready.append(cleaned)
            self.buffer = ""

        return ready


def play_tts_audio(audio, sample_rate):
    device_info = sd.query_devices(TTS_OUTPUT_DEVICE, "output")
    max_channels = int(device_info["max_output_channels"])
    if max_channels < 1:
        raise RuntimeError("Output device has no channels.")

    channel_options = [2, 1] if max_channels >= 2 else [1]
    device_rate = int(round(device_info["default_samplerate"]))
    rate_options = [int(sample_rate)]
    if device_rate > 0 and device_rate not in rate_options:
        rate_options.append(device_rate)

    last_error = None
    for output_rate in rate_options:
        if output_rate == sample_rate:
            converted = np.asarray(audio, dtype=np.float32)
        else:
            divisor = gcd(int(sample_rate), output_rate)
            converted = resample_poly(
                audio, output_rate // divisor, int(sample_rate) // divisor
            ).astype(np.float32)

        for channels in channel_options:
            output = (
                np.column_stack((converted, converted))
                if channels == 2 else converted.reshape(-1, 1)
            )
            try:
                with sd.OutputStream(
                    device=TTS_OUTPUT_DEVICE,
                    samplerate=output_rate,
                    channels=channels,
                    dtype="float32",
                    latency="high",
                ) as stream:
                    stream.write(np.ascontiguousarray(output, dtype=np.float32))
                return output_rate, channels, len(output) / output_rate
            except Exception as error:
                last_error = error

    raise RuntimeError(f"Audio playback failed: {last_error}")


def tts_worker(voice):
    while True:
        text = tts_queue.get()
        if text is None:
            break

        text = clean_text_for_tts(text)
        if not text:
            continue

        tts_result_queue.put(("status", "Synthesizing..."))
        try:
            chunks = list(voice.synthesize(text))
            if not chunks:
                raise RuntimeError("Piper returned no audio.")

            audio = np.concatenate([
                chunk.audio_float_array for chunk in chunks
            ]).astype(np.float32)
            sample_rate = int(chunks[0].sample_rate)
            if audio.size == 0 or sample_rate <= 0:
                raise RuntimeError("Invalid Piper audio.")

            clear_audio_queue()
            tts_playing.set()
            tts_result_queue.put(("status", "Speaking"))
            started = time.perf_counter()
            try:
                rate, channels, duration = play_tts_audio(audio, sample_rate)
            finally:
                tts_playing.clear()
                clear_audio_queue()

            tts_result_queue.put((
                "done",
                {
                    "duration": duration,
                    "elapsed": time.perf_counter() - started,
                    "rate": rate,
                    "channels": channels,
                },
            ))
        except Exception as error:
            tts_playing.clear()
            clear_audio_queue()
            tts_result_queue.put(("error", f"{type(error).__name__}: {error}"))


def process_tts_results(state, history):
    while True:
        try:
            kind, value = tts_result_queue.get_nowait()
        except queue.Empty:
            break

        if kind == "status":
            state["tts_status"] = value
        elif kind == "done":
            state["tts_status"] = (
                f"Ready | {value['channels']}ch | {value['duration']:.1f}s"
            )
        else:
            state["tts_status"] = "Playback error"
            history.append((datetime.now().strftime("%H:%M:%S"), f"TTS error: {value}"))


# ============================================================
# Noise calibration and VAD
# ============================================================

def threshold_from_noise(noise_floor):
    """Convert measured background RMS into a bounded speech gate."""
    return min(
        MAX_ENERGY_THRESHOLD,
        max(MIN_FRAME_RMS, float(noise_floor) * NOISE_MULTIPLIER),
    )


def frame_rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    audio = np.nan_to_num(audio, copy=False)
    # Remove DC offset before measuring microphone energy.
    audio = audio - float(np.mean(audio))
    return float(np.sqrt(np.mean(audio * audio)))


def calibrate_noise(input_rate, progress_callback=None):
    """Measure ambient noise while the user remains silent."""
    levels = []
    target = int(CALIBRATION_SECONDS * 1000 / FRAME_MS)
    deadline = time.monotonic() + CALIBRATION_SECONDS + 8

    while len(levels) < target:
        if time.monotonic() > deadline:
            raise RuntimeError("Noise calibration timed out; check the microphone.")
        try:
            block = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio = resample_audio(block, input_rate)
        expected = SAMPLE_RATE * FRAME_MS // 1000
        if len(audio) != expected:
            continue

        levels.append(frame_rms(audio))
        if progress_callback:
            progress_callback(len(levels) / target)

    # Median is less affected by a short click or sudden background sound.
    noise = float(np.percentile(levels, 40))
    threshold = threshold_from_noise(noise)
    return noise, threshold


class VoiceSegmenter:
    """Adaptive WebRTC VAD with noise tracking and explicit silence hangover."""

    def __init__(self, input_rate, energy_threshold, noise_floor=0.0):
        self.input_rate = input_rate
        self.noise_floor = max(float(noise_floor), MIN_FRAME_RMS)
        self.energy_threshold = max(float(energy_threshold), MIN_FRAME_RMS)
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.pre_roll = deque(maxlen=max(1, PRE_ROLL_MS // FRAME_MS))
        self.start_window = deque(maxlen=max(1, START_WINDOW_MS // FRAME_MS))
        self.sequence = 0
        self.last_rms = 0.0
        self.speech_rms = 0.0
        self.adaptive_noise = True
        self.reset()

    def reset(self):
        self.frames = []
        self.voiced_frames = 0
        self.silence_frames = 0
        self.last_endpoint_quiet = True
        self.active = False
        self.current_id = None
        self.last_partial_time = 0.0
        self.speech_rms = 0.0
        self.pre_roll.clear()
        self.start_window.clear()

    def set_noise(self, noise_floor, threshold=None, adaptive=True):
        self.noise_floor = max(float(noise_floor), MIN_FRAME_RMS)
        self.adaptive_noise = adaptive
        self.energy_threshold = (
            threshold_from_noise(self.noise_floor)
            if threshold is None else max(float(threshold), MIN_FRAME_RMS)
        )

    def detect_speech(self, frame):
        audio = resample_audio(frame, self.input_rate)
        expected = SAMPLE_RATE * FRAME_MS // 1000
        if len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))
        else:
            audio = audio[:expected]

        audio = np.nan_to_num(np.clip(audio, -1.0, 1.0), copy=False)
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(audio * audio)))
        self.last_rms = rms

        pcm = (audio * 32767).astype("<i2").tobytes()
        try:
            vad_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
        except Exception:
            vad_speech = False

        # Separate speech onset from continuation to avoid clipping soft syllables.
        start_gate = max(self.energy_threshold, self.noise_floor * NOISE_MULTIPLIER)
        continue_gate = max(
            MIN_FRAME_RMS, self.noise_floor * CONTINUE_NOISE_MULTIPLIER
        )
        speech_gate = continue_gate if self.active else start_gate
        speech = bool(vad_speech and rms >= speech_gate)

        if speech:
            if self.speech_rms <= 0.0:
                self.speech_rms = rms
            else:
                self.speech_rms = 0.90 * self.speech_rms + 0.10 * rms

        # Endpointing uses both the calibrated room floor and the recent speech level.
        end_gate = max(
            2.0 * MIN_FRAME_RMS,
            self.noise_floor * END_NOISE_MULTIPLIER,
            self.speech_rms * END_SPEECH_RATIO,
        )
        self.last_endpoint_quiet = (not vad_speech) or (rms < end_gate)

        # Track slow room-noise changes only while idle; never learn from speech.
        if (self.adaptive_noise and not self.active and not vad_speech
                and rms < max(self.noise_floor * 2.2, 5.0 * MIN_FRAME_RMS)):
            self.noise_floor = (
                (1.0 - NOISE_UPDATE_ALPHA) * self.noise_floor
                + NOISE_UPDATE_ALPHA * rms
            )
            self.energy_threshold = threshold_from_noise(self.noise_floor)
        return speech

    def finish(self):
        if not self.active or not self.frames:
            self.reset()
            return None

        uid = self.current_id
        duration_ms = len(self.frames) * FRAME_MS
        audio = np.concatenate(self.frames).astype(np.float32)
        valid = (
            duration_ms >= MIN_UTTERANCE_MS
            and self.voiced_frames >= MIN_VOICED_FRAMES
        )
        self.reset()
        return (uid, audio) if valid else None

    def push(self, frame):
        speech = self.detect_speech(frame)

        if not self.active:
            self.pre_roll.append((frame.copy(), speech))
            self.start_window.append(speech)
            required = max(5, ceil(len(self.start_window) * START_RATIO))

            if (
                len(self.start_window) == self.start_window.maxlen
                and sum(self.start_window) >= required
            ):
                self.sequence += 1
                self.current_id = self.sequence
                self.active = True
                self.frames = [item[0] for item in self.pre_roll]
                self.voiced_frames = sum(1 for item in self.pre_roll if item[1])
                self.silence_frames = 0
                self.last_partial_time = time.monotonic()
                return "started", self.current_id, None
            return None, None, None

        self.frames.append(frame.copy())
        if speech:
            self.voiced_frames += 1

        # Count low-energy tail silence independently from voiced-frame count.
        if self.last_endpoint_quiet or not speech:
            self.silence_frames += 1
        else:
            self.silence_frames = 0

        duration = len(self.frames) * FRAME_MS / 1000
        silence_ms = self.silence_frames * FRAME_MS
        if silence_ms >= END_SILENCE_MS or duration >= MAX_UTTERANCE_SECONDS:
            final = self.finish()
            if final is None:
                return "discarded", None, None
            uid, audio = final
            return "final", uid, audio

        now = time.monotonic()
        if (
            duration >= PARTIAL_MIN_SECONDS
            and self.voiced_frames >= MIN_VOICED_FRAMES
            and now - self.last_partial_time >= PARTIAL_INTERVAL
        ):
            self.last_partial_time = now
            return "partial", self.current_id, np.concatenate(self.frames).astype(np.float32)
        return None, None, None


# ============================================================
# Speech recognition worker
# ============================================================

def asr_worker(recognizer, input_rate):
    while True:
        task = asr_queue.get()
        if task is None:
            break

        kind, uid, audio = task
        try:
            audio = prepare_asr_audio(resample_audio(audio, input_rate))
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)

            started = time.perf_counter()
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - started

            text = stream.result.text.strip()
            duration = len(audio) / SAMPLE_RATE
            result_queue.put((kind, uid, text, duration, elapsed))
        except Exception as error:
            result_queue.put(("error", uid, str(error), 0, 0))


# ============================================================
# Ollama worker and chat submission
# ============================================================

def configure_ollama_connection(url, use_proxy=False):
    """Never send loopback Ollama traffic through HTTP(S)_PROXY."""
    host = (urlsplit(url).hostname or "").lower()
    local_host = host in {"127.0.0.1", "localhost", "::1"}
    if local_host or not use_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def check_ollama_connection():
    """Return a short local Ollama status without using system proxies."""
    parsed = urlsplit(OLLAMA_URL)
    base = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(base + "/api/version", method="GET")
    try:
        with OLLAMA_OPENER.open(request, timeout=3) as response:
            info = json.loads(response.read().decode("utf-8", errors="replace"))
        return f"Connected ({info.get('version', 'version unknown')})"
    except Exception as error:
        return f"Unavailable: {type(error).__name__}: {error}"


def ollama_worker():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_text = chat_queue.get()
        if user_text is None:
            break

        request_messages = messages + [{"role": "user", "content": user_text}]
        payload = {
            "model": OLLAMA_MODEL,
            "messages": request_messages,
            "stream": True,
            "options": {"temperature": 0.4, "num_ctx": 4096},
        }

        parts = []
        speech_buffer = StreamingSpeechBuffer(words_per_chunk=TTS_STREAM_WORDS)
        started = time.monotonic()
        chunks = 0

        try:
            request = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
                method="POST",
            )
            chat_result_queue.put(("start", None))

            with OLLAMA_OPENER.open(request, timeout=OLLAMA_TIMEOUT) as response:
                while not cancel_event.is_set():
                    if time.monotonic() - started > OLLAMA_TIMEOUT:
                        raise TimeoutError(f"Ollama timed out after {OLLAMA_TIMEOUT}s")
                    try:
                        raw = response.readline()
                    except socket.timeout:
                        continue

                    if not raw:
                        break
                    if not raw.strip():
                        continue

                    packet = json.loads(raw.decode("utf-8"))
                    piece = packet.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        chunks += 1
                        chat_result_queue.put((
                            "chunk",
                            (piece, chunks, time.monotonic() - started),
                        ))
                        # Start Piper while Qwen is still generating its answer.
                        if TTS_STREAMING_ENABLED and not cancel_event.is_set():
                            for phrase in speech_buffer.feed(piece):
                                tts_queue.put(phrase)

                    if packet.get("error"):
                        raise RuntimeError(packet["error"])
                    if packet.get("done"):
                        break

            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", "پاسخ هوش مصنوعی لغو شد."))
                continue

            answer = "".join(parts).strip()
            if not answer:
                raise RuntimeError("Ollama returned an empty response.")

            if TTS_STREAMING_ENABLED:
                for phrase in speech_buffer.feed(final=True):
                    tts_queue.put(phrase)

            messages = request_messages + [{"role": "assistant", "content": answer}]
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            chat_result_queue.put((
                "done",
                (answer, chunks, time.monotonic() - started),
            ))

        except Exception as error:
            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", "پاسخ هوش مصنوعی لغو شد."))
            else:
                detail = f"{type(error).__name__}: {error}"
                if isinstance(error, urllib.error.URLError):
                    detail += " | Ollama local URL: " + OLLAMA_URL
                    detail += " | Localhost proxy bypass is enabled."
                chat_result_queue.put(("error", detail))


def submit_text(value, state, history):
    """Send typed or recognized text through the same chat queue."""
    value = value.strip()
    if not value:
        return False

    if state["ai_busy"]:
        return False

    timestamp = datetime.now().strftime("%H:%M:%S")
    history.append((timestamp, f"شما: {value}"))

    # Save accepted typed and recognized messages.
    try:
        with OUTPUT_FILE.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {value}\n")
    except OSError as error:
        history.append((timestamp, f"Save error: {error}"))

    cancel_event.clear()
    state["ai_busy"] = True
    state["ai_status"] = "Waiting for Ollama"
    state["stream_text"] = ""
    state["stream_chunks"] = 0
    state["stream_started"] = time.monotonic()
    state["scroll_offset"] = 0
    chat_queue.put(value)
    return True


def process_chat_results(state, history):
    while True:
        try:
            kind, value = chat_result_queue.get_nowait()
        except queue.Empty:
            break

        if kind == "start":
            state["ai_busy"] = True
            state["ai_status"] = "Generating response"
            state["stream_text"] = ""
            state["stream_chunks"] = 0

        elif kind == "chunk":
            piece, count, elapsed = value
            state["stream_text"] += piece
            state["stream_chunks"] = count
            rate = count / max(elapsed, 0.001)
            state["ai_status"] = f"Streaming | {rate:.1f} chunks/s"

        elif kind == "done":
            answer, count, elapsed = value
            timestamp = datetime.now().strftime("%H:%M:%S")
            history.append((timestamp, f"Qwen: {answer}"))
            state["ai_busy"] = False
            state["ai_status"] = f"Completed | {elapsed:.1f}s"
            state["stream_text"] = ""
            state["scroll_offset"] = 0


        elif kind == "cancelled":
            history.append((datetime.now().strftime("%H:%M:%S"), value))
            state["ai_busy"] = False
            state["ai_status"] = "Cancelled"
            state["stream_text"] = ""
            state["stream_chunks"] = 0
            state["scroll_offset"] = 0

        elif kind == "error":
            history.append((
                datetime.now().strftime("%H:%M:%S"),
                f"Ollama error: {value}",
            ))
            state["ai_busy"] = False
            state["ai_status"] = "Ollama error"
            state["stream_text"] = ""
            state["message"] = value


# ============================================================
# ASR result handling
# ============================================================

def process_results(state, history):
    while True:
        try:
            kind, uid, text, duration, elapsed = result_queue.get_nowait()
        except queue.Empty:
            break

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
                # Submit recognized speech through the shared chat path.
                if state["ai_busy"]:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    history.append((timestamp, f"شما: {text}"))
                    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
                        file.write(f"[{timestamp}] {text}\n")
                    history.append((timestamp, "سیستم: Qwen مشغول بود؛ گفتار ارسال نشد."))
                else:
                    submit_text(text, state, history)

                state["last_rtf"] = elapsed / duration if duration else 0
                state["last_inference"] = elapsed

            if state["active_id"] == uid:
                state["active_id"] = None
                state["partial_text"] = ""
                state["speaking"] = False
                state["status"] = "Listening"


# ============================================================
# Keyboard input
# ============================================================

_KEY_BUFFER = bytearray()
_ESC_PENDING_AT = None


def _decode_terminal_key(buffer):
    """Decode one UTF-8 character or a complete ANSI key/mouse sequence."""
    if not buffer:
        return None, 0

    if buffer[0] == 0x1B:
        if len(buffer) == 1:
            return None, 0

        # X10 mouse: ESC [ M followed by three binary bytes.
        if buffer.startswith(b"\x1b[M"):
            if len(buffer) < 6:
                return None, 0
            button = max(0, buffer[3] - 32)
            if button & 64:
                return ("up" if (button & 1) == 0 else "down"), 6
            return None, 6

        if buffer[1] == ord('['):
            # CSI ends with a byte in 0x40..0x7E.
            final_index = 2
            while final_index < len(buffer) and not 0x40 <= buffer[final_index] <= 0x7E:
                final_index += 1
            if final_index >= len(buffer):
                return None, 0

            body = bytes(buffer[2:final_index]).decode("ascii", "ignore")
            final = chr(buffer[final_index])
            consumed = final_index + 1

            # SGR mouse; wheel reports must never become text characters.
            if body.startswith("<") and final in ("M", "m"):
                try:
                    button = int(body[1:].split(";", 1)[0])
                except ValueError:
                    return None, consumed
                if button & 64:
                    return ("up" if (button & 1) == 0 else "down"), consumed
                return None, consumed

            if final in "ABCDHF":
                return {
                    "A": "up", "B": "down", "C": "right", "D": "left",
                    "H": "home", "F": "end",
                }[final], consumed

            if final == "~":
                code = body.split(";", 1)[0]
                return {
                    "1": "home", "7": "home", "4": "end", "8": "end",
                    "3": "delete", "5": "pageup", "6": "pagedown",
                    "200": None, "201": None,
                }.get(code), consumed

            return None, consumed

        if buffer[1] == ord('O'):
            if len(buffer) < 3:
                return None, 0
            key = chr(buffer[2])
            return {"A": "up", "B": "down", "C": "right", "D": "left",
                    "H": "home", "F": "end"}.get(key), 3

        # Alt/unknown escape sequence: consume ESC only; do not leak its bytes.
        return "esc", 1

    first = buffer[0]
    if first in (10, 13):
        return "enter", 1
    if first in (8, 127):
        return "backspace", 1
    if first == 21:
        return "clear", 1
    if first == 3:
        return "ctrl_c", 1

    # Decode exactly one UTF-8 code point, even when bytes arrive separately.
    if first < 0x80:
        size = 1
    elif first & 0xE0 == 0xC0:
        size = 2
    elif first & 0xF0 == 0xE0:
        size = 3
    elif first & 0xF8 == 0xF0:
        size = 4
    else:
        return "�", 1

    if len(buffer) < size:
        return None, 0
    try:
        return bytes(buffer[:size]).decode("utf-8"), size
    except UnicodeDecodeError:
        return "�", size


def read_key():
    """Read UTF-8 safely and consume mouse/ANSI events without leaking bytes."""
    global _ESC_PENDING_AT
    if not sys.stdin.isatty():
        return None

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], 0)
    if ready:
        try:
            _KEY_BUFFER.extend(os.read(fd, 256))
        except OSError:
            return None

    while _KEY_BUFFER:
        key, consumed = _decode_terminal_key(_KEY_BUFFER)
        if consumed:
            del _KEY_BUFFER[:consumed]
            _ESC_PENDING_AT = None
            if key is not None:
                return key
            continue

        # A lone ESC is a key only after a short wait for sequence bytes.
        if _KEY_BUFFER == b"\x1b":
            now = time.monotonic()
            if _ESC_PENDING_AT is None:
                _ESC_PENDING_AT = now
                return None
            if now - _ESC_PENDING_AT >= 0.06:
                _KEY_BUFFER.clear()
                _ESC_PENDING_AT = None
                return "esc"
        return None
    return None


def list_microphones():
    console.print("\n[bold cyan]Available microphones[/bold cyan]\n")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            console.print(
                f"{index}: {device['name']} ({device['default_samplerate']} Hz)"
            )


# ============================================================
# Main application
# ============================================================

def main():
    global OLLAMA_URL, OLLAMA_MODEL, OLLAMA_OPENER, TTS_STREAMING_ENABLED
    parser = argparse.ArgumentParser(description="Shenava Persian voice/text chat")
    parser.add_argument("--device", type=int, default=INPUT_DEVICE)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help="Ollama /api/chat URL (default: local server)")
    parser.add_argument("--model", default=OLLAMA_MODEL,
                        help="Ollama model name")
    parser.add_argument("--ollama-use-proxy", action="store_true",
                        help="Use environment proxy only for a non-local Ollama URL")
    args = parser.parse_args()

    OLLAMA_URL = args.ollama_url
    OLLAMA_MODEL = args.model
    OLLAMA_OPENER = configure_ollama_connection(OLLAMA_URL, args.ollama_use_proxy)

    if args.list_devices:
        list_microphones()
        return

    # Use UTF-8 streams when supported by the terminal's Python runtime.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    try:
        locale.setlocale(locale.LC_CTYPE, "")
    except locale.Error:
        pass

    console.print("[cyan]Loading Shenava...[/cyan]")
    console.print("[dim]Ollama localhost traffic bypasses system proxies.[/dim]")
    console.print(f"[dim]Chat model: {model_display_name()}[/dim]")
    recognizer = load_model()

    tts_voice = None
    tts_thread = None
    if TTS_ENABLED_BY_DEFAULT and not args.no_tts:
        tts_voice = load_tts_model()
        if tts_voice is not None:
            tts_thread = threading.Thread(
                target=tts_worker, args=(tts_voice,),
                daemon=True, name="Piper-TTS",
            )
            tts_thread.start()
            TTS_STREAMING_ENABLED = True

    device = sd.query_devices(args.device, "input")
    input_rate = int(round(device["default_samplerate"]))
    frame_samples = round(input_rate * FRAME_MS / 1000)

    segmenter = VoiceSegmenter(input_rate, MIN_FRAME_RMS, MIN_FRAME_RMS)
    history = deque(maxlen=VISIBLE_HISTORY)

    state = {
        "status": "Starting",
        "device": device["name"],
        "threshold": MIN_FRAME_RMS,
        "noise_floor": MIN_FRAME_RMS,
        "silence_ms": 0,
        "calibrated": False,
        "speaking": False,
        "partial_text": "",
        "active_id": None,
        "message": "",
        "last_rtf": 0.0,
        "last_inference": 0.0,
        "scroll_offset": 0,
        "ai_busy": False,
        "ai_status": check_ollama_connection(),
        "stream_text": "",
        "stream_chunks": 0,
        "stream_started": 0.0,
        "input_text": "",
        "input_cursor": 0,
        "tts_enabled": tts_thread is not None,
        "tts_status": "Ready" if tts_thread else "Disabled",
    }

    asr_thread = threading.Thread(
        target=asr_worker, args=(recognizer, input_rate),
        daemon=True, name="Sherpa-ASR",
    )
    chat_thread = threading.Thread(
        target=ollama_worker, daemon=True, name="Ollama-chat",
    )
    asr_thread.start()
    chat_thread.start()

    old_terminal = None
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_terminal = termios.tcgetattr(fd)
        # Cbreak keeps terminal input responsive while preserving Unicode.
        tty.setcbreak(fd)

    try:
        try:
            sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l")
            sys.stdout.flush()
        except OSError:
            pass

        with Live(
            make_layout(state, history),
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
                    state["status"] = "Calibrating — stay silent"
                    clear_audio_queue()

                    def calibration_progress(progress):
                        state["status"] = (
                            f"Calibrating {progress * 100:.0f}% — stay silent"
                        )
                        live.update(make_layout(state, history), refresh=True)

                    noise, threshold = calibrate_noise(
                        input_rate, calibration_progress
                    )
                    segmenter.set_noise(noise, threshold)
                    state["noise_floor"] = noise
                    state["threshold"] = threshold
                    state["calibrated"] = True
                    state["status"] = "Listening"
                    clear_audio_queue()

                    while True:
                        process_results(state, history)
                        process_chat_results(state, history)
                        process_tts_results(state, history)

                        if state["ai_busy"] and not state["stream_chunks"]:
                            waited = time.monotonic() - state["stream_started"]
                            state["ai_status"] = f"Waiting for first token | {waited:.1f}s"

                        key = read_key()
                        current_input = state["input_text"]

                        if key == "ctrl_c":
                            break

                        elif key == "enter":
                            if current_input.strip():
                                accepted = submit_text(current_input, state, history)
                                # Keep the text if Qwen is busy.
                                if accepted:
                                    state["input_text"] = ""
                                    state["input_cursor"] = 0

                        elif key == "backspace":
                            pos = state.get("input_cursor", len(current_input))
                            if pos > 0:
                                state["input_text"] = current_input[:pos - 1] + current_input[pos:]
                                state["input_cursor"] = pos - 1

                        elif key == "delete":
                            pos = state.get("input_cursor", len(current_input))
                            if pos < len(current_input):
                                state["input_text"] = current_input[:pos] + current_input[pos + 1:]

                        elif key == "clear":
                            state["input_text"] = ""
                            state["input_cursor"] = 0

                        elif key == "left":
                            state["input_cursor"] = max(0, state.get("input_cursor", len(current_input)) - 1)
                        elif key == "right":
                            state["input_cursor"] = min(len(current_input), state.get("input_cursor", len(current_input)) + 1)
                        elif key == "home":
                            state["input_cursor"] = 0
                        elif key == "end":
                            state["input_cursor"] = len(current_input)

                        elif key == "esc":
                            if state["ai_busy"]:
                                cancel_event.set()
                                state["ai_status"] = "Cancelling..."

                        elif key == "up":
                            state["scroll_offset"] += 1
                        elif key == "down":
                            state["scroll_offset"] = max(0, state["scroll_offset"] - 1)
                        elif key == "pageup":
                            state["scroll_offset"] += max(1, console.height - 20)
                        elif key == "pagedown":
                            state["scroll_offset"] = max(
                                0, state["scroll_offset"] - max(1, console.height - 20)
                            )
                        # Commands are active only when the text field is empty.
                        elif not current_input and key == "x":
                            if state["ai_busy"]:
                                cancel_event.set()
                        elif not current_input and key == "c":
                            final = segmenter.finish()
                            if final:
                                asr_queue.put(("final", *final))

                            state["speaking"] = False
                            state["partial_text"] = ""
                            state["status"] = "Calibrating — stay silent"
                            clear_audio_queue()

                            noise, threshold = calibrate_noise(
                                input_rate, calibration_progress
                            )
                            segmenter.set_noise(noise, threshold)
                            state["noise_floor"] = noise
                            state["threshold"] = threshold
                            state["calibrated"] = True
                            state["status"] = "Listening"
                            clear_audio_queue()
                            continue
                        elif not current_input and key == "d":
                            segmenter.set_noise(MIN_FRAME_RMS, MIN_FRAME_RMS, adaptive=False)
                            state["noise_floor"] = MIN_FRAME_RMS
                            state["threshold"] = MIN_FRAME_RMS
                            state["calibrated"] = False
                            state["status"] = "Listening"
                            clear_audio_queue()
                        elif not current_input and key == "q":
                            break

                        # Append printable Unicode input (including Persian).
                        elif (
                            isinstance(key, str)
                            and len(key) == 1
                            and key.isprintable()
                        ):
                            pos = state.get("input_cursor", len(current_input))
                            state["input_text"] = current_input[:pos] + key + current_input[pos:]
                            state["input_cursor"] = pos + len(key)

                        try:
                            block = audio_queue.get(timeout=0.05)
                        except queue.Empty:
                            live.update(make_layout(state, history))
                            continue

                        event, uid, audio = segmenter.push(block)
                        state["threshold"] = segmenter.energy_threshold
                        state["noise_floor"] = segmenter.noise_floor
                        state["silence_ms"] = (
                            segmenter.silence_frames * FRAME_MS
                            if segmenter.active else 0
                        )

                        if event == "started":
                            state["active_id"] = uid
                            state["speaking"] = True
                            state["partial_text"] = ""
                            state["status"] = "Speech detected"

                        elif event == "partial":
                            state["speaking"] = True
                            if asr_queue.qsize() < 3:
                                asr_queue.put_nowait(("partial", uid, audio))

                        elif event == "final":
                            state["status"] = "Finalizing"
                            state["speaking"] = False
                            state["silence_ms"] = 0
                            asr_queue.put(("final", uid, audio))

                        elif event == "discarded":
                            state["active_id"] = None
                            state["partial_text"] = ""
                            state["speaking"] = False
                            state["silence_ms"] = 0
                            state["status"] = "Listening"

                        live.update(make_layout(state, history))

            except KeyboardInterrupt:
                state["status"] = "Stopping"

            finally:
                remaining = segmenter.finish()
                if remaining:
                    asr_queue.put(("final", *remaining))

                # Stop ASR after queued recognition tasks.
                asr_queue.put(None)
                asr_thread.join()
                process_results(state, history)

                # Stop the active chat request and its worker.
                cancel_event.set()
                chat_queue.put(None)
                chat_thread.join(timeout=3.0)
                process_chat_results(state, history)

                # Stop TTS gracefully.
                if tts_thread is not None:
                    # Drain pending speech before stopping the Piper worker.
                    tts_queue.put(None)
                    tts_thread.join(timeout=5.0)
                    process_tts_results(state, history)

                state["status"] = "Stopped"
                state["speaking"] = False
                state["partial_text"] = ""
                live.update(make_layout(state, history), refresh=True)

    finally:
        if old_terminal is not None:
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, old_terminal
            )
        # Reset common xterm mouse-reporting modes in case they were enabled.
        try:
            sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l")
            sys.stdout.flush()
        except OSError:
            pass

        console.print(f"\n[green]Transcript saved:[/green] {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()