#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shenava Persian voice + text chat (Sherpa-ONNX, Ollama, Piper)."""

import argparse
import json
import queue
import re
import select
import socket
import sys
import termios
import textwrap
import threading
import time
import tty
import urllib.request
from collections import deque
from datetime import datetime
from math import ceil, gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import sherpa_onnx
from scipy.signal import resample_poly
from huggingface_hub import snapshot_download
import arabic_reshaper
from bidi.algorithm import get_display

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
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_TIMEOUT = 300

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
TTS_QUEUE_SIZE = 3

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20

VAD_MODE = 3
START_WINDOW_MS = 500
START_RATIO = 0.60
PRE_ROLL_MS = 300
END_SILENCE_MS = 900

CALIBRATION_SECONDS = 2
MIN_FRAME_RMS = 0.0001
NOISE_MULTIPLIER = 2.5

MIN_VOICED_FRAMES = 8
MIN_UTTERANCE_MS = 500
MAX_UTTERANCE_SECONDS = 20

PARTIAL_INTERVAL = 1.0
PARTIAL_MIN_SECONDS = 1.0
CPU_THREADS = 4
INPUT_DEVICE = None

AUDIO_QUEUE_SIZE = 500
VISIBLE_HISTORY = 50

console = Console(highlight=False)

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
asr_queue = queue.Queue()
result_queue = queue.Queue()
chat_queue = queue.Queue()
chat_result_queue = queue.Queue()
tts_queue = queue.Queue(maxsize=TTS_QUEUE_SIZE)
tts_result_queue = queue.Queue()
cancel_event = threading.Event()
tts_playing = threading.Event()
audio_dropped = [0]


# ============================================================
# Persian display helpers
# ============================================================

def rtl(value):
    if not value:
        return ""
    return get_display(arabic_reshaper.reshape(str(value)), base_dir="R")


def rtl_renderable(value, width=None, style="white"):
    if width is None:
        width = max(10, console.width - 8)

    lines = []
    for paragraph in str(value).splitlines() or [""]:
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        lines.extend(rtl(line) for line in wrapped)

    return Group(*[
        Align.right(Text(line, style=style, no_wrap=True, overflow="crop"))
        for line in lines
    ])


# ============================================================
# Terminal interface
# ============================================================

def make_layout(state, history):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=7),
        Layout(name="transcript", ratio=1, minimum_size=7),
        Layout(name="input", size=3),
        Layout(name="footer", size=3),
    )

    header = Text()
    header.append(" SHENAVA ", style="bold bright_cyan")
    header.append("VOICE + TEXT CHAT", style="bold white")
    header.append(f"     KOOCHIK FP32 | {OLLAMA_MODEL}", style="dim")
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
    table.add_row("VAD", "WebRTC", "CALIBRATION", calibration)

    ai_style = "bold cyan" if state["ai_busy"] else "dim"
    table.add_row("AI", Text(state["ai_status"], style=ai_style),
                  "MODEL", OLLAMA_MODEL)

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

    # Build transcript lines before applying the viewport.
    width = max(12, console.width - 8)
    transcript = []

    for timestamp, message in history:
        transcript.append((f"  {timestamp}", "dim", False))
        for paragraph in str(message).splitlines() or [""]:
            wrapped = textwrap.wrap(
                paragraph, width=width, break_long_words=True,
                break_on_hyphens=False
            ) or [""]
            transcript.extend((rtl(line), "white", True) for line in wrapped)

    if state["speaking"]:
        transcript.append(("  LIVE PREVIEW", "bold yellow", False))
        preview = state["partial_text"] or "در حال شنیدن گفتار..."
        for paragraph in preview.splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, width=width) or [""]
            transcript.extend(
                (rtl(line), "bold bright_white", True) for line in wrapped
            )

    if state["ai_busy"]:
        transcript.append(("  QWEN STREAMING", "bold bright_cyan", False))
        stream = state["stream_text"] or "در انتظار پاسخ..."
        for paragraph in stream.splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, width=width) or [""]
            transcript.extend(
                (rtl(line), "bright_white", True) for line in wrapped
            )

    if not transcript:
        transcript.append((rtl("هنوز پیامی ثبت نشده است."), "dim", True))

    viewport = max(1, console.height - 20)
    max_offset = max(0, len(transcript) - viewport)
    offset = min(max(0, state["scroll_offset"]), max_offset)
    state["scroll_offset"] = offset
    end = len(transcript) - offset
    visible = transcript[max(0, end - viewport):end]

    rendered = []
    for line, style, is_rtl in visible:
        item = Text(line, style=style, no_wrap=True, overflow="crop")
        rendered.append(Align.right(item, pad=True) if is_rtl else item)

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
    prompt = state["input_text"] or "اینجا تایپ کنید..."
    layout["input"].update(
        Panel(
            rtl_renderable(prompt + "  ", width=max(10, console.width - 10)),
            title="[bold cyan]TEXT INPUT[/bold cyan]",
            border_style="cyan", padding=(0, 1),
        )
    )

    footer = Text(
        "ENTER Send | BACKSPACE Delete | CTRL+U Clear | ESC Cancel | "
        "↑/↓ Scroll | C Calibrate | D Disable | Q Quit",
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
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=["model.onnx", "tokens.txt", "persian_itn.py"],
        )

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


def audio_callback(indata, frames, time_info, status):
    # Ignore microphone input while Piper is speaking.
    if tts_playing.is_set():
        return

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
        value = tts_queue.get()
        if value is None:
            break

        text = clean_text_for_tts(value)
        if not text:
            tts_result_queue.put(("status", "Ready"))
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

def calibrate_noise(input_rate, progress_callback=None):
    levels = []
    target = int(CALIBRATION_SECONDS * 1000 / FRAME_MS)
    deadline = time.monotonic() + CALIBRATION_SECONDS + 5

    while len(levels) < target:
        if time.monotonic() > deadline:
            raise RuntimeError("Noise calibration timed out.")

        try:
            block = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio = resample_audio(block, input_rate)
        expected = SAMPLE_RATE * FRAME_MS // 1000
        if len(audio) != expected:
            continue

        levels.append(float(np.sqrt(np.mean(audio ** 2))))
        if progress_callback:
            progress_callback(len(levels) / target)

    noise = float(np.percentile(levels, 75))
    return noise, max(MIN_FRAME_RMS, noise * NOISE_MULTIPLIER)


class VoiceSegmenter:
    def __init__(self, input_rate, energy_threshold):
        self.input_rate = input_rate
        self.energy_threshold = energy_threshold
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.pre_roll = deque(maxlen=PRE_ROLL_MS // FRAME_MS)
        self.start_window = deque(maxlen=START_WINDOW_MS // FRAME_MS)
        self.reset()
        self.sequence = 0

    def reset(self):
        self.frames = []
        self.voiced_frames = 0
        self.silence_frames = 0
        self.active = False
        self.current_id = None
        self.last_partial_time = 0.0
        self.pre_roll.clear()
        self.start_window.clear()

    def detect_speech(self, frame):
        audio = resample_audio(frame, self.input_rate)
        expected = SAMPLE_RATE * FRAME_MS // 1000
        if len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))
        else:
            audio = audio[:expected]

        audio = np.clip(audio, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.energy_threshold:
            return False

        pcm = (audio * 32767).astype("<i2").tobytes()
        return self.vad.is_speech(pcm, SAMPLE_RATE)

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
            required = ceil(len(self.start_window) * START_RATIO)

            if (
                len(self.start_window) == self.start_window.maxlen
                and sum(self.start_window) >= required
            ):
                self.sequence += 1
                self.current_id = self.sequence
                self.active = True
                self.frames = [item[0] for item in self.pre_roll]
                self.voiced_frames = sum(item[1] for item in self.pre_roll)
                self.silence_frames = 0
                self.last_partial_time = time.monotonic()
                return "started", self.current_id, None

            return None, None, None

        self.frames.append(frame.copy())
        if speech:
            self.voiced_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

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
            audio = np.clip(resample_audio(audio, input_rate), -1.0, 1.0)
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
        started = time.monotonic()
        chunks = 0

        try:
            request = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            chat_result_queue.put(("start", None))

            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
                while not cancel_event.is_set():
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
                chat_result_queue.put(("error", str(error)))


def submit_text(value, state, history):
    """Send typed or recognized text through the same chat queue."""
    value = value.strip()
    if not value:
        return False

    timestamp = datetime.now().strftime("%H:%M:%S")
    history.append((timestamp, f"شما: {value}"))

    # Save both typed and recognized messages.
    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {value}\n")

    if state["ai_busy"]:
        history.append((timestamp, "سیستم: Qwen مشغول است؛ پیام جدید پذیرفته نشد."))
        return False

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

            if state["tts_enabled"]:
                try:
                    tts_queue.put_nowait(answer)
                except queue.Full:
                    tts_result_queue.put(("error", "TTS queue is full."))

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

def read_key():
    """Read one Unicode character or a terminal control sequence."""
    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None

    char = sys.stdin.read(1)

    if char == "\x1b":
        sequence = char
        for _ in range(5):
            ready, _, _ = select.select([sys.stdin], [], [], 0.015)
            if not ready:
                return "esc"
            sequence += sys.stdin.read(1)
            if sequence.endswith(("A", "B", "H", "F", "~")):
                return {
                    "\x1b[A": "up",
                    "\x1b[B": "down",
                    "\x1b[5~": "pageup",
                    "\x1b[6~": "pagedown",
                    "\x1b[F": "end",
                    "\x1b[H": "home",
                }.get(sequence)
        return None

    if char in ("\r", "\n"):
        return "enter"
    if char in ("\x7f", "\x08"):
        return "backspace"
    if char == "\x15":
        return "clear"
    if char == "\x03":
        return "ctrl_c"
    return char


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
    parser = argparse.ArgumentParser(description="Shenava Persian voice/text chat")
    parser.add_argument("--device", type=int, default=INPUT_DEVICE)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
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

    segmenter = VoiceSegmenter(input_rate, MIN_FRAME_RMS)
    history = deque(maxlen=VISIBLE_HISTORY)

    state = {
        "status": "Starting",
        "device": device["name"],
        "threshold": MIN_FRAME_RMS,
        "calibrated": False,
        "speaking": False,
        "partial_text": "",
        "active_id": None,
        "message": "",
        "last_rtf": 0.0,
        "last_inference": 0.0,
        "scroll_offset": 0,
        "ai_busy": False,
        "ai_status": "Idle",
        "stream_text": "",
        "stream_chunks": 0,
        "stream_started": 0.0,
        "input_text": "",
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

                    _, threshold = calibrate_noise(
                        input_rate, calibration_progress
                    )
                    segmenter.energy_threshold = threshold
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

                        elif key == "backspace":
                            state["input_text"] = current_input[:-1]

                        elif key == "clear":
                            state["input_text"] = ""

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
                        elif key in ("end",):
                            state["scroll_offset"] = 0
                        elif key == "home":
                            state["scroll_offset"] = 10**9

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

                            _, threshold = calibrate_noise(
                                input_rate, calibration_progress
                            )
                            segmenter.energy_threshold = threshold
                            state["threshold"] = threshold
                            state["calibrated"] = True
                            state["status"] = "Listening"
                            clear_audio_queue()
                            continue
                        elif not current_input and key == "d":
                            segmenter.energy_threshold = MIN_FRAME_RMS
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
                            state["input_text"] += key

                        try:
                            block = audio_queue.get(timeout=0.05)
                        except queue.Empty:
                            live.update(make_layout(state, history))
                            continue

                        event, uid, audio = segmenter.push(block)

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
                            asr_queue.put(("final", uid, audio))

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
                    try:
                        tts_queue.put_nowait(None)
                    except queue.Full:
                        while True:
                            try:
                                tts_queue.get_nowait()
                            except queue.Empty:
                                break
                        tts_queue.put_nowait(None)
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

        console.print(f"\n[green]Transcript saved:[/green] {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
