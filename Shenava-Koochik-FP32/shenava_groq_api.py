#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""Shenava Persian voice + text chat (Sherpa-ONNX, Groq API, Piper)."""



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
import urllib.error

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

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_TIMEOUT = 300

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# urllib uses HTTP_PROXY / HTTPS_PROXY from the environment for this cloud API.

GROQ_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies()))



SYSTEM_PROMPT = (

    "طبیعی، دقیق و کوتاه پاسخ بده. فقط متن ساده تولید کن. "

    "هرگز از ایموجی، صورتک متنی، تصویر، نشانه‌گذاری تصویر Markdown، "

    "تگ HTML تصویر یا لینک تصویر استفاده نکن؛ فقط پاسخ متنی بده."

)



BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "shenava-koochik-fp32"

MODEL_FILE = MODEL_DIR / "model.onnx"

TOKENS_FILE = MODEL_DIR / "tokens.txt"

OUTPUT_FILE = BASE_DIR / "transcriptions.txt"



TTS_MODEL_FILE = BASE_DIR / "models" / "fa_IR-ganji-medium.onnx"

TTS_OUTPUT_DEVICE = None

TTS_ENABLED_BY_DEFAULT = True

TTS_QUEUE_SIZE = 0  # Unbounded queue for complete assistant replies



SAMPLE_RATE = 16000

CHANNELS = 1

FRAME_MS = 20



# Aggressive mode reduces false speech detections from steady room noise.

VAD_MODE = 2                    # Balanced: less likely to reject quiet speech.

START_WINDOW_MS = 400

START_RATIO = 0.45

PRE_ROLL_MS = 360

END_SILENCE_MS = 950



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



console = Console(highlight=False, soft_wrap=True, emoji=False)



audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

asr_queue = queue.Queue()

result_queue = queue.Queue()

chat_queue = queue.Queue()

chat_result_queue = queue.Queue()

tts_queue = queue.Queue(maxsize=TTS_QUEUE_SIZE)

tts_result_queue = queue.Queue()

cancel_event = threading.Event()

tts_playing = threading.Event()

mic_paused = threading.Event()

audio_dropped = [0]

audio_status = [""]





# ============================================================

# Persian display helpers

# Keep Unicode unchanged; directional marks only set the paragraph direction.

# ============================================================



def rtl(value):

    """Keep original letters and tell the terminal this line is RTL."""

    return "\u200f" + str(value) if value is not None else ""





# Remove visual-only content from model output; keep ordinary text untouched.

_EMOJI_RE = re.compile(

    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"

    "\uFE0E\uFE0F\u200D\u20E3\U0001F3FB-\U0001F3FF]"

)

_TEXT_EMOTICON_RE = re.compile(
    r"(?<!\w)(?::|;|=|8|x|X)[-^']?[)(/DPp]|<3|¯\\_\(ツ\)_/¯(?!\w)"
)


def clean_model_output(value, streaming=False):

    """Strip emoji, emoticons and image markup from assistant text."""

    text = str(value or "")



    # Do not expose unfinished image syntax while tokens are still arriving.

    if streaming:

        lower = text.lower()

        open_markers = [

            (lower.rfind("!["), ")"),

            (lower.rfind("<img"), ">"),

            (lower.rfind("<svg"), "</svg>"),

            (lower.rfind("<picture"), "</picture>"),

            (lower.rfind("<figure"), "</figure>"),

        ]

        for position, closing in open_markers:

            if position >= 0 and closing not in lower[position:]:

                text = text[:position]

                lower = text.lower()



    # Remove Markdown images, HTML image containers and embedded image data.
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"!\[[^\]]*\](?!\()", " ", text)
    text = re.sub(r"<img\b[^>]*?/?>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<(picture|figure|svg)\b.*?</\1\s*>", " ", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", " ", text)
    text = re.sub(
        r"https?://[^\s)]+\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s)]*)?",
        " ", text, flags=re.IGNORECASE,
    )

    # Unicode emoji and common ASCII-style faces.

    text = _EMOJI_RE.sub("", text)

    text = _TEXT_EMOTICON_RE.sub("", text)

    text = re.sub(r"(?<!\w):[a-z0-9_+-]{2,}:(?!\w)", "", text, flags=re.IGNORECASE)

    return text.strip()





def ltr(value):

    """Keep timestamps and numeric labels left-to-right."""

    return "\u200e" + str(value)





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

        Layout(name="transcript", ratio=1, minimum_size=7),

        Layout(name="input", size=3),

        Layout(name="footer", size=2),

    )



    header = Text()

    header.append(" SHENAVA ", style="bold bright_cyan")

    header.append("VOICE + TEXT CHAT", style="bold white")

    status = state.get("status", "Listening").upper()

    header.append(f"     {status}", style="bold green" if status == "LISTENING" else "bold yellow")

    layout["header"].update(

        Panel(Align.center(header), box=box.DOUBLE,

              border_style="cyan", padding=(0, 1))

    )



    # Only completed turns appear here; partial ASR and token streams stay hidden.

    width = max(12, console.width - 10)

    transcript = []



    for timestamp, message in history:

        transcript.append((ltr(f"  {timestamp}"), "dim", False))

        # The model ID is part of the first response line, not a separate row.

        append_rtl_lines(transcript, message, width, "white")



    if not transcript:

        transcript.append((ltr("Start speaking or type a message."), "dim", False))



    viewport = max(1, console.height - 12)

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

        rendered.append(

            Align.right(item, pad=True) if is_rtl else Align.left(item, pad=True)

        )



    layout["transcript"].update(

        Panel(

            Group(*rendered),

            title="[bold green]CHAT[/bold green]",

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

            title="[bold cyan]MESSAGE[/bold cyan]",

            border_style="cyan", padding=(0, 1),

        )

    )



    footer = Text(

        "ENTER Send | ↑/↓ Scroll | ←/→ Edit | ESC Cancel | Q Quit",

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

    # Freeze microphone capture for the complete assistant turn.

    if status:

        audio_status[0] = str(status)

    if mic_paused.is_set() or tts_playing.is_set():

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





def clear_tts_queue():

    """Drop queued speech that belongs to an interrupted assistant turn."""

    while True:

        try:

            tts_queue.get_nowait()

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
    value = clean_model_output(value, streaming=False)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"(?m)^\s*[-*+]\s+", "", value)
    value = re.sub(r"[*_#>~]", "", value)
    return re.sub(r"\s+", " ", value).strip()



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

        task = tts_queue.get()

        if task is None:

            break



        # This marker is queued after the final phrase; it confirms playback ended.

        if isinstance(task, tuple) and task[0] == "response_end":

            tts_result_queue.put(("response_done", task[1]))

            continue



        text = task



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

            if value == "Speaking":

                state["status"] = "Speaking"

        elif kind == "done":

            pass

        elif kind == "response_done":

            state["tts_completed_turn"] = value

            finish_turn_if_ready(state, value)

        else:

            # Keep the chat clean; expose playback failure only in the status state.

            state["message"] = str(value)





def finish_turn_if_ready(state, turn_id):

    """Resume microphone only after Groq and the final Piper audio both finish."""

    if turn_id != state.get("turn_id") or not state.get("assistant_done"):

        return

    if state.get("tts_enabled") and state.get("tts_completed_turn") != turn_id:

        state["waiting_for_tts"] = True

        state["status"] = "Speaking"

        return



    state["waiting_for_tts"] = False

    state["ai_busy"] = False

    state["status"] = "Listening"

    mic_paused.clear()

    clear_audio_queue()





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

# Groq worker and chat submission

# ============================================================



def iter_groq_stream_text(payload, stop_event=None):
    """Yield assistant text from Groq's Server-Sent Events response."""
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Shenava/1.0",
        },
        method="POST",
    )

    with GROQ_OPENER.open(request, timeout=GROQ_TIMEOUT) as response:
        data_lines = []
        deadline = time.monotonic() + GROQ_TIMEOUT

        while stop_event is None or not stop_event.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Groq timed out after {GROQ_TIMEOUT}s")

            try:
                raw = response.readline()
            except socket.timeout:
                continue

            if not raw:
                if data_lines:
                    event_data = "\n".join(data_lines).strip()
                    if event_data and event_data != "[DONE]":
                        yield _groq_packet_text(event_data)
                return

            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

            # SSE events end with a blank line; comments and metadata are ignored.
            if not line:
                if not data_lines:
                    continue
                event_data = "\n".join(data_lines).strip()
                data_lines.clear()
                if not event_data:
                    continue
                if event_data == "[DONE]":
                    return
                piece = _groq_packet_text(event_data)
                if piece:
                    yield piece
                continue

            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


def _groq_packet_text(event_data):
    """Extract only user-facing content; never expose reasoning fields."""
    try:
        packet = json.loads(event_data)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid Groq stream event: {error}") from error

    if packet.get("error"):
        api_error = packet["error"]
        if isinstance(api_error, dict):
            api_error = api_error.get("message", api_error)
        raise RuntimeError(str(api_error))

    choices = packet.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def test_groq_connection():
    """Test the same streaming request path used by the chat worker."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": "سلام. فقط بگو Groq works!"}
        ],
        "stream": True,
        "max_tokens": 100,
    }

    console.print(f"[cyan]Testing Groq streaming API: {GROQ_MODEL}[/cyan]")
    answer_parts = []
    try:
        for piece in iter_groq_stream_text(payload):
            answer_parts.append(piece)
        answer = clean_model_output("".join(answer_parts)).strip()
        if not answer:
            raise RuntimeError("Groq connected but returned no assistant text.")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            api_error = parsed.get("error", {})
            detail = api_error.get("message", body) if isinstance(api_error, dict) else str(api_error)
        except (json.JSONDecodeError, AttributeError):
            detail = body or str(error)
        raise RuntimeError(f"Groq API HTTP {error.code}: {detail}") from error

    console.print("[bold green]Groq API connected; streaming test passed.[/bold green]")
    console.print(f"[dim]Model: {GROQ_MODEL}[/dim]")
    console.print(f"[green]Response: {answer}[/green]")


def groq_worker():
    """Generate complete assistant replies through Groq's streaming Chat API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        task = chat_queue.get()
        if task is None:
            break

        if isinstance(task, tuple):
            user_text, turn_id = task
        else:
            user_text, turn_id = task, 0

        request_messages = messages + [{"role": "user", "content": user_text}]
        payload = {
            "model": GROQ_MODEL,
            "messages": request_messages,
            "stream": True,
            # The proven curl request used max_tokens; keep the same API parameter.
            "max_tokens": 1024,
        }

        parts = []
        started = time.monotonic()
        try:
            for piece in iter_groq_stream_text(payload, cancel_event):
                if piece:
                    parts.append(piece)

            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", turn_id))
                continue

            answer = clean_model_output("".join(parts)).strip()
            if not answer:
                raise RuntimeError("Groq returned an empty response.")

            # Preserve the in-memory conversation history for this session.
            messages = request_messages + [{"role": "assistant", "content": answer}]
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            chat_result_queue.put((
                "done",
                (turn_id, answer, time.monotonic() - started),
            ))

        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                api_error = parsed.get("error", {})
                detail = api_error.get("message", body) if isinstance(api_error, dict) else str(api_error)
            except (json.JSONDecodeError, AttributeError):
                detail = body or str(error)

            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", turn_id))
            else:
                chat_result_queue.put((
                    "error", (turn_id, f"Groq API HTTP {error.code}: {detail}")
                ))

        except Exception as error:
            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", turn_id))
            else:
                detail = f"{type(error).__name__}: {error}"
                if isinstance(error, urllib.error.URLError):
                    detail += " | Check HTTP_PROXY / HTTPS_PROXY settings."
                chat_result_queue.put(("error", (turn_id, detail)))



def submit_text(value, state, history):

    """Send typed or recognized text through the same chat queue."""

    value = value.strip()

    if not value:

        return False



    if state["ai_busy"] or state.get("asr_pending", False):

        return False



    timestamp = datetime.now().strftime("%H:%M:%S")

    history.append((timestamp, f"You: {value}"))



    # Save accepted typed and recognized messages.

    try:

        with OUTPUT_FILE.open("a", encoding="utf-8") as file:

            file.write(f"[{timestamp}] {value}\n")

    except OSError as error:

        state["message"] = f"Could not save transcript: {error}"



    cancel_event.clear()

    state["turn_id"] = state.get("turn_id", 0) + 1

    state["ai_busy"] = True

    state["assistant_done"] = False

    state["waiting_for_tts"] = False

    state["asr_pending"] = False

    state["message"] = ""
    state["status"] = "Thinking"

    state["scroll_offset"] = 0

    mic_paused.set()

    clear_audio_queue()

    chat_queue.put((value, state["turn_id"]))

    return True





def process_chat_results(state, history):

    while True:

        try:

            kind, value = chat_result_queue.get_nowait()

        except queue.Empty:

            break



        if kind == "done":

            turn_id, answer, elapsed = value

            if turn_id != state.get("turn_id"):

                continue



            timestamp = datetime.now().strftime("%H:%M:%S")

            history.append((timestamp, f"Groq: {answer}"))

            state["scroll_offset"] = 0

            state["assistant_done"] = True



            # The full answer is rendered first; Piper is queued after the UI refresh.

            if state.get("tts_enabled"):

                state["tts_pending"] = (turn_id, answer)

                state["status"] = "Preparing speech"

                state["waiting_for_tts"] = True

            else:

                finish_turn_if_ready(state, turn_id)



        elif kind == "cancelled":

            if value != state.get("turn_id"):

                continue

            state["assistant_done"] = True

            state["tts_completed_turn"] = value

            finish_turn_if_ready(state, value)



        elif kind == "error":

            turn_id, detail = value

            if turn_id != state.get("turn_id"):

                continue

            state["message"] = detail
            history.append((
                datetime.now().strftime("%H:%M:%S"),
                f"System: Groq request failed: {detail[:180]}",
            ))
            state["assistant_done"] = True

            state["tts_completed_turn"] = turn_id

            finish_turn_if_ready(state, turn_id)





def enqueue_pending_tts_after_render(state):

    """Queue the complete answer only after its final text is drawn on screen."""

    pending = state.pop("tts_pending", None)

    if not pending:

        return



    turn_id, answer = pending

    if turn_id != state.get("turn_id") or not state.get("tts_enabled"):

        return



    # Queue order guarantees response_done arrives after all audio is played.

    tts_queue.put(answer)

    tts_queue.put(("response_end", turn_id))





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

            state["message"] = text

            state["active_id"] = None

            state["speaking"] = False

            state["asr_pending"] = False

            state["status"] = "Listening"

            mic_paused.clear()

            clear_audio_queue()

            continue



        if kind == "partial":

            # Partial hypotheses are intentionally hidden to keep chat stable.

            continue



        if kind == "final":

            state["asr_pending"] = False

            if text:

                submit_text(text, state, history)



                state["last_rtf"] = elapsed / duration if duration else 0

                state["last_inference"] = elapsed

            else:

                # Empty recognition must not leave the microphone locked.

                mic_paused.clear()



            if state["active_id"] == uid:

                state["active_id"] = None

                state["partial_text"] = ""

                state["speaking"] = False

                if not state["ai_busy"]:

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

    global GROQ_MODEL, GROQ_API_KEY, GROQ_OPENER

    parser = argparse.ArgumentParser(description="Shenava Persian voice/text chat")

    parser.add_argument("--device", type=int, default=INPUT_DEVICE)

    parser.add_argument("--list-devices", action="store_true")

    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--test-groq", action="store_true", help="Test Groq API and exit")

    parser.add_argument("--model", default=GROQ_MODEL,

                        help="Groq model ID")

    args = parser.parse_args()



    GROQ_MODEL = args.model

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

    GROQ_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies()))



    if args.list_devices:

        list_microphones()

        return



    if not GROQ_API_KEY:

        raise SystemExit(

            "GROQ_API_KEY is not set. Set it with: export GROQ_API_KEY=YOUR_KEY"

        )



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



    if args.test_groq:
        try:
            test_groq_connection()
        except Exception as error:
            console.print(f"[bold red]Groq test failed:[/bold red] {type(error).__name__}: {error}")
            raise SystemExit(1) from error
        return

    console.print("[cyan]Loading Shenava...[/cyan]")

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

        "turn_id": 0,

        "assistant_done": False,

        "waiting_for_tts": False,

        "tts_completed_turn": 0,

        "asr_pending": False,

        "input_text": "",

        "input_cursor": 0,

        "tts_enabled": tts_thread is not None,

        "tts_pending": None,

    }



    asr_thread = threading.Thread(

        target=asr_worker, args=(recognizer, input_rate),

        daemon=True, name="Sherpa-ASR",

    )

    chat_thread = threading.Thread(

        target=groq_worker, daemon=True, name="Groq-chat",

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



                        key = read_key()

                        current_input = state["input_text"]



                        if key == "ctrl_c":

                            break



                        elif key == "enter":

                            if current_input.strip():

                                accepted = submit_text(current_input, state, history)

                                # Keep the text if Groq is busy.

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

                            enqueue_pending_tts_after_render(state)

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



                        elif event == "partial":

                            state["speaking"] = True



                        elif event == "final":

                            # Freeze the microphone before final ASR begins.

                            mic_paused.set()

                            clear_audio_queue()

                            state["asr_pending"] = True

                            state["status"] = "Processing"

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

                        enqueue_pending_tts_after_render(state)



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