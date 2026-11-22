
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
import json
import textwrap
import urllib.request
import urllib.error
import socket

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

# Ollama chat API
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "ibnsina/ibnsina-3b:latest"
OLLAMA_TIMEOUT = 300  # Allow  to load and start generating on CPU
SYSTEM_PROMPT = (
    "تو یک دستیار فارسی‌زبان هستی. طبیعی، دقیق و کوتاه پاسخ بده. "
    "به زبان کاربر پاسخ بده و اگر چیزی را نمی‌دانی صادقانه بگو."
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "shenava-koochik-fp32"

MODEL_FILE = MODEL_DIR / "model.onnx"
TOKENS_FILE = MODEL_DIR / "tokens.txt"

OUTPUT_FILE = BASE_DIR / "transcriptions.txt"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20

# VAD
VAD_MODE = 3
START_WINDOW_MS = 500
START_RATIO = 0.60

PRE_ROLL_MS = 300
END_SILENCE_MS = 900

# Noise calibration
CALIBRATION_SECONDS = 2
MIN_FRAME_RMS = 0.0001
NOISE_MULTIPLIER = 2.5

MIN_VOICED_FRAMES = 8
MIN_UTTERANCE_MS = 500
MAX_UTTERANCE_SECONDS = 20

# Live partial recognition
PARTIAL_INTERVAL = 1.0
PARTIAL_MIN_SECONDS = 1.0

# CPU
CPU_THREADS = 4

# Microphone
INPUT_DEVICE = None

AUDIO_QUEUE_SIZE = 500
VISIBLE_HISTORY = 50

# ============================================================

console = Console(highlight=False)

audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
asr_queue = queue.Queue()
result_queue = queue.Queue()
chat_queue = queue.Queue()
chat_result_queue = queue.Queue()
cancel_event = threading.Event()

audio_dropped = [0]


# ============================================================
# Persian RTL
# ============================================================

def rtl(text):

    if not text:
        return ""

    return get_display(
        arabic_reshaper.reshape(text),
        base_dir="R"
    )


def rtl_renderable(text, style="white", width=None):
    """Wrap Persian text in logical order, then shape each line separately.

    Rich must not wrap an already bidi-reordered paragraph: doing so reverses
    the visual order of wrapped lines. We therefore wrap first and disable
    Rich's second wrapping pass.
    """
    if not text:
        return Align.right(Text("", style=style))

    if width is None:
        # Panel border + horizontal padding leave a few columns unavailable.
        width = max(10, console.width - 8)

    visual_lines = []
    for paragraph in str(text).splitlines() or [str(text)]:
        if not paragraph:
            visual_lines.append("")
            continue

        logical_lines = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        visual_lines.extend(rtl(line) for line in logical_lines)

    # Align each visual line independently. A single multi-line Text wrapped
    # in Align.right may align only the first line correctly in Rich.
    aligned_lines = [
        Align.right(
            Text(
                line,
                style=style,
                no_wrap=True,
                overflow="crop",
            ),
            pad=True,
        )
        for line in visual_lines
    ]

    return Group(*aligned_lines)


# ============================================================
# Fixed terminal layout
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
    header.append("VOICE CHAT", style="bold white")
    header.append(f"     KOOCHIK FP32  |  {OLLAMA_MODEL}", style="dim")
    layout["header"].update(Panel(Align.center(header), box=box.DOUBLE,
                                  border_style="cyan", padding=(0, 1)))

    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="dim")
    status_table.add_column(style="bold white")
    status_table.add_column(style="dim")
    status_table.add_column(style="bold cyan")
    status_style = "bold green" if state["status"] == "Listening" else "bold yellow"
    status_table.add_row("STATUS", Text(state["status"], style=status_style),
                         "MICROPHONE", state["device"])
    calibration_text = (f"ACTIVE  ({state['threshold']:.4f})" if state["calibrated"]
                        else f"DISABLED ({state['threshold']:.4f})")
    status_table.add_row("VAD", "WebRTC", "CALIBRATION", calibration_text)
    ai_state = state.get("ai_status", "Idle")
    ai_style = "bold cyan" if state.get("ai_busy") else "dim"
    status_table.add_row("AI", Text(ai_state, style=ai_style), "MODEL", OLLAMA_MODEL)
    layout["status"].update(Panel(status_table, title="[bold cyan]SYSTEM STATUS[/bold cyan]",
                                  border_style="cyan", box=box.ROUNDED))

    # Build a physical-line buffer first. Scroll offset is measured from the
    # bottom: 0 means follow newest messages; positive values show older lines.
    width = max(12, console.width - 8)
    transcript_lines = []
    for timestamp, message in history:
        transcript_lines.append((f"  {timestamp}", "dim", False))
        for paragraph in str(message).splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True,
                                    break_on_hyphens=False, replace_whitespace=False,
                                    drop_whitespace=True) or [""]
            for line in wrapped:
                transcript_lines.append((rtl(line), "white", True))

    if state["speaking"]:
        transcript_lines.append(("  LIVE PREVIEW", "bold yellow", False))
        preview = state["partial_text"] or "در حال شنیدن گفتار..."
        for paragraph in preview.splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True,
                                    break_on_hyphens=False, replace_whitespace=False,
                                    drop_whitespace=True) or [""]
            for line in wrapped:
                transcript_lines.append((rtl(line), "bold bright_white", True))
    if state.get("ai_busy"):
        transcript_lines.append((" STREAMING", "bold bright_cyan", False))
        stream_text = state.get("stream_text", "") or "در انتظار اولین بخش پاسخ..."
        for paragraph in stream_text.splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True,
                                    break_on_hyphens=False, replace_whitespace=False,
                                    drop_whitespace=True) or [""]
            for line in wrapped:
                transcript_lines.append((rtl(line), "bright_white", True))
    elif not transcript_lines:
        transcript_lines.append((rtl("هنوز گفتاری ثبت نشده است."), "dim", True))

    # Account for the transcript panel border and its vertical padding.
    viewport = max(1, console.height - 16)
    max_offset = max(0, len(transcript_lines) - viewport)
    offset = min(max(0, state.get("scroll_offset", 0)), max_offset)
    state["scroll_offset"] = offset
    end = len(transcript_lines) - offset
    begin = max(0, end - viewport)
    visible_lines = transcript_lines[begin:end]

    renderables = []
    for line, style, is_rtl in visible_lines:
        item = Text(line, style=style, no_wrap=True, overflow="crop")
        renderables.append(Align.right(item, pad=True) if is_rtl else item)

    scroll_label = "LIVE" if offset == 0 else f"SCROLL: {offset} lines up"
    transcript_panel = Panel(
        Group(*renderables),
        title="[bold green]TRANSCRIPT[/bold green]",
        subtitle=f"[dim]{scroll_label} | ↑/↓: scroll | PgUp/PgDn: page | End: latest[/dim]",
        border_style="green", box=box.ROUNDED, padding=(1, 2),
    )
    layout["transcript"].update(transcript_panel)

    footer = Text(
        " Speak to chat    X / ESC Cancel AI     ↑/↓ Scroll     PgUp/PgDn Page     C Recalibrate     D Disable     Q Quit",
        style="bold white", justify="center",
    )
    layout["footer"].update(Panel(footer, border_style="blue", box=box.ROUNDED,
                                  padding=(0, 1)))
    return layout


# ============================================================
# Model
# ============================================================

def load_model():

    if not MODEL_FILE.exists() or not TOKENS_FILE.exists():

        console.print(
            "[yellow]Downloading Koochik FP32...[/yellow]"
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=[
                "model.onnx",
                "tokens.txt",
                "persian_itn.py",
            ],
        )

    console.print(
        "[cyan]Initializing Koochik FP32...[/cyan]"
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(MODEL_FILE),
        tokens=str(TOKENS_FILE),
        num_threads=CPU_THREADS,
        decoding_method="greedy_search",
        debug=False,
    )

    console.print(
        "[bold green]Model loaded successfully.[/bold green]"
    )

    return recognizer


# ============================================================
# Audio utilities
# ============================================================

def resample_audio(audio, input_rate):

    if input_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)

    divisor = gcd(input_rate, SAMPLE_RATE)

    return np.asarray(
        resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            input_rate // divisor,
        ),
        dtype=np.float32,
    )


def audio_callback(indata, frames, time_info, status):

    try:

        audio_queue.put_nowait(
            indata[:, 0].copy()
        )

    except queue.Full:

        audio_dropped[0] += 1

        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(
                indata[:, 0].copy()
            )
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

    deadline = (
        time.monotonic()
        + CALIBRATION_SECONDS
        + 5
    )

    while len(levels) < target_frames:

        if time.monotonic() > deadline:
            raise RuntimeError(
                "Noise calibration timed out."
            )

        try:

            block = audio_queue.get(timeout=0.1)

        except queue.Empty:
            continue

        audio = resample_audio(
            block,
            input_rate,
        )

        expected = SAMPLE_RATE * FRAME_MS // 1000

        if len(audio) != expected:
            continue

        rms = float(
            np.sqrt(np.mean(audio ** 2))
        )

        levels.append(rms)

        if progress_callback:

            progress_callback(
                len(levels) / target_frames
            )

    noise_rms = float(
        np.percentile(levels, 75)
    )

    threshold = max(
        MIN_FRAME_RMS,
        noise_rms * NOISE_MULTIPLIER,
    )

    return noise_rms, threshold


# ============================================================
# Voice segmenter with partial results
# ============================================================

class VoiceSegmenter:

    def __init__(self, input_rate, energy_threshold):

        self.input_rate = input_rate
        self.energy_threshold = energy_threshold

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

    def reset(self):

        self.pre_roll.clear()
        self.start_window.clear()
        self.frames.clear()

        self.voiced_frames = 0
        self.silence_frames = 0

        self.active = False
        self.current_id = None

        self.last_partial_time = 0.0

    def detect_speech(self, frame):

        audio = resample_audio(
            frame,
            self.input_rate,
        )

        expected = SAMPLE_RATE * FRAME_MS // 1000

        if len(audio) < expected:

            audio = np.pad(
                audio,
                (0, expected - len(audio)),
            )

        elif len(audio) > expected:

            audio = audio[:expected]

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        rms = float(
            np.sqrt(np.mean(audio ** 2))
        )

        if rms < self.energy_threshold:
            return False

        pcm = (
            audio * 32767
        ).astype("<i2").tobytes()

        return self.vad.is_speech(
            pcm,
            SAMPLE_RATE,
        )

    def finish(self):

        if not self.active or not self.frames:
            self.reset()
            return None

        uid = self.current_id

        duration_ms = len(self.frames) * FRAME_MS

        audio = np.concatenate(
            self.frames,
        ).astype(np.float32)

        valid = (
            duration_ms >= MIN_UTTERANCE_MS
            and self.voiced_frames >= MIN_VOICED_FRAMES
        )

        self.reset()

        if not valid:
            return None

        return uid, audio

    def push(self, frame):

        speech = self.detect_speech(frame)

        # Waiting for speech
        if not self.active:

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
                    item[0]
                    for item in self.pre_roll
                ]

                self.voiced_frames = sum(
                    item[1]
                    for item in self.pre_roll
                )

                self.silence_frames = 0

                self.last_partial_time = time.monotonic()

                return "started", self.current_id, None

            return None, None, None

        # Active speech
        self.frames.append(frame.copy())

        if speech:

            self.voiced_frames += 1
            self.silence_frames = 0

        else:

            self.silence_frames += 1

        duration = len(self.frames) * FRAME_MS / 1000

        silence_ms = self.silence_frames * FRAME_MS

        # Final segment
        if (
            silence_ms >= END_SILENCE_MS
            or duration >= MAX_UTTERANCE_SECONDS
        ):

            final = self.finish()

            if final is None:
                return "discarded", None, None

            uid, audio = final

            return "final", uid, audio

        # Live partial recognition
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
# ASR worker
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
# Ollama chat worker
# ============================================================

def ollama_worker():
    """Run one Ollama request at a time and stream chunks; supports cancellation."""
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
        answer_parts = []
        started = time.monotonic()
        chunk_count = 0
        cancelled = False

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
                        raw_line = response.readline()
                    except socket.timeout:
                        continue
                    if not raw_line:
                        break
                    if not raw_line.strip():
                        continue
                    packet = json.loads(raw_line.decode("utf-8"))
                    piece = packet.get("message", {}).get("content", "")
                    if piece:
                        answer_parts.append(piece)
                        chunk_count += 1
                        elapsed = max(0.001, time.monotonic() - started)
                        chat_result_queue.put(("chunk", (piece, chunk_count, elapsed)))
                    if packet.get("error"):
                        raise RuntimeError(packet["error"])
                    if packet.get("done"):
                        break

            cancelled = cancel_event.is_set()
            if cancelled:
                chat_result_queue.put(("cancelled", "پاسخ هوش مصنوعی توسط کاربر لغو شد."))
                continue

            answer = "".join(answer_parts).strip()
            if not answer:
                raise RuntimeError("Ollama returned an empty response.")

            messages = request_messages + [{"role": "assistant", "content": answer}]
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            elapsed = max(0.001, time.monotonic() - started)
            chat_result_queue.put(("done", (answer, chunk_count, elapsed)))

        except Exception as error:
            if cancel_event.is_set():
                chat_result_queue.put(("cancelled", "پاسخ هوش مصنوعی توسط کاربر لغو شد."))
            else:
                chat_result_queue.put(("error", str(error)))


def process_chat_results(state, history):
    changed = False
    while True:
        try:
            kind, value = chat_result_queue.get_nowait()
        except queue.Empty:
            break

        changed = True
        if kind == "start":
            state["ai_busy"] = True
            state["ai_status"] = "Generating response..."
            state["stream_text"] = ""
            state["stream_chunks"] = 0
            state["stream_started"] = time.monotonic()

        elif kind == "chunk":
            piece, chunk_count, elapsed = value
            state["stream_text"] += piece
            state["stream_chunks"] = chunk_count
            rate = chunk_count / max(elapsed, 0.001)
            state["ai_status"] = f"Streaming | {rate:.1f} chunks/s | {elapsed:.1f}s"

        elif kind == "done":
            answer, chunk_count, elapsed = value
            timestamp = datetime.now().strftime("%H:%M:%S")
            history.append((timestamp, f"ابن سینا: {answer}"))
            state["ai_busy"] = False
            state["ai_status"] = f"Completed | {elapsed:.1f}s | {chunk_count} chunks"
            state["stream_text"] = ""
            state["scroll_offset"] = 0

        elif kind == "cancelled":
            timestamp = datetime.now().strftime("%H:%M:%S")
            history.append((timestamp, value))
            state["ai_busy"] = False
            state["ai_status"] = "Cancelled"
            state["stream_text"] = ""
            state["stream_chunks"] = 0
            state["scroll_offset"] = 0

        elif kind == "error":
            timestamp = datetime.now().strftime("%H:%M:%S")
            history.append((timestamp, f"خطای Ollama: {value}"))
            state["ai_busy"] = False
            state["ai_status"] = "Ollama error"
            state["stream_text"] = ""
            state["message"] = value

    return changed


# ============================================================
# Result handling and file saving
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

                timestamp = datetime.now().strftime(
                    "%H:%M:%S"
                )

                # Save original Unicode Persian text.
                with OUTPUT_FILE.open(
                    "a",
                    encoding="utf-8",
                ) as file:

                    file.write(
                        f"[{timestamp}] {text}\n"
                    )
                    file.flush()

                history.append(
                    (timestamp, f"شما: {text}")
                )

                # Accept only one user turn at a time. Speech recognized while
                #  is busy is shown in the transcript but is not queued.
                if state.get("ai_busy"):
                    state["ai_status"] = "Busy - wait or press X to cancel"
                    history.append((timestamp, "سیستم: تا پایان پاسخ فعلی، پرسش جدید پذیرفته نمی‌شود."))
                else:
                    cancel_event.clear()
                    state["ai_busy"] = True
                    state["ai_status"] = "Queued / waiting for Ollama"
                    state["stream_text"] = ""
                    state["stream_chunks"] = 0
                    state["stream_started"] = time.monotonic()
                    chat_queue.put(text)

                state["last_rtf"] = (
                    elapsed / duration if duration else 0
                )

                state["last_inference"] = elapsed

            if state["active_id"] == uid:

                state["active_id"] = None
                state["partial_text"] = ""
                state["speaking"] = False
                state["status"] = "Listening"


# ============================================================
# Keyboard
# ============================================================

def read_key():
    """Read normal keys and common terminal navigation keys."""
    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None

    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch.lower()

    # Arrow / PageUp / PageDown / End escape sequence.
    sequence = ch
    for _ in range(5):
        ready, _, _ = select.select([sys.stdin], [], [], 0.015)
        if not ready:
            if sequence == "\x1b":
                return "esc"
            break
        sequence += sys.stdin.read(1)
        if sequence.endswith(("A", "B", "H", "F", "~")):
            if sequence.endswith("[A"):
                return "up"
            if sequence.endswith("[B"):
                return "down"
            if sequence.endswith("[5~"):
                return "pageup"
            if sequence.endswith("[6~"):
                return "pagedown"
            if sequence.endswith("[F") or sequence.endswith("[4~"):
                return "end"
            if sequence.endswith("[H") or sequence.endswith("[1~"):
                return "home"
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

    show_header = console.print

    console.print(
        "[cyan]Loading Koochik FP32...[/cyan]"
    )

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

    segmenter = VoiceSegmenter(
        input_rate,
        MIN_FRAME_RMS,
    )

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
    }

    worker = threading.Thread(
        target=asr_worker,
        args=(recognizer, input_rate),
        daemon=True,
    )

    worker.start()

    chat_worker = threading.Thread(
        target=ollama_worker,
        daemon=True,
    )
    chat_worker.start()

    old_terminal = None

    if sys.stdin.isatty():

        fd = sys.stdin.fileno()

        old_terminal = termios.tcgetattr(fd)

        tty.setcbreak(fd)

    stop_worker = False

    try:

        initial_layout = make_layout(state, history)

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

                    # Initial calibration
                    state["status"] = "Calibrating — stay silent"

                    live.update(
                        make_layout(state, history),
                        refresh=True,
                    )

                    clear_audio_queue()

                    def calibration_progress(progress):

                        state["status"] = (
                            f"Calibrating {progress * 100:.0f}% — stay silent"
                        )

                        live.update(
                            make_layout(state, history),
                            refresh=True,
                        )

                    noise_rms, threshold = calibrate_noise(
                        input_rate,
                        calibration_progress,
                    )

                    segmenter.energy_threshold = threshold

                    state["threshold"] = threshold
                    state["calibrated"] = True
                    state["status"] = "Listening"
                    state["message"] = ""

                    clear_audio_queue()

                    while True:

                        process_results(
                            state,
                            history,
                        )
                        process_chat_results(state, history)
                        if state.get("ai_busy") and not state.get("stream_chunks"):
                            waited = time.monotonic() - state.get("stream_started", time.monotonic())
                            state["ai_status"] = f"Waiting for first token | {waited:.1f}s"

                        key = read_key()

                        # Cancel the active Ollama generation without stopping Shenava.
                        if key in ("x", "esc") and state.get("ai_busy"):
                            cancel_event.set()
                            state["ai_status"] = "Cancelling..."

                        # Transcript navigation (offset 0 follows newest messages).
                        if key in ("up", "k"):
                            state["scroll_offset"] += 1
                        elif key in ("down", "j"):
                            state["scroll_offset"] = max(0, state["scroll_offset"] - 1)
                        elif key == "pageup":
                            state["scroll_offset"] += max(1, console.height - 18)
                        elif key == "pagedown":
                            state["scroll_offset"] = max(0, state["scroll_offset"] - max(1, console.height - 18))
                        elif key in ("end", "g"):
                            state["scroll_offset"] = 0
                        elif key == "home":
                            state["scroll_offset"] = 10**9

                        # Recalibrate
                        if key == "c":

                            final = segmenter.finish()

                            if final is not None:

                                uid, audio = final

                                asr_queue.put(
                                    ("final", uid, audio)
                                )

                            state["status"] = "Calibrating — stay silent"
                            state["speaking"] = False
                            state["partial_text"] = ""

                            clear_audio_queue()

                            live.update(
                                make_layout(state, history),
                                refresh=True,
                            )

                            noise_rms, threshold = calibrate_noise(
                                input_rate,
                                calibration_progress,
                            )

                            segmenter.energy_threshold = threshold

                            state["threshold"] = threshold
                            state["calibrated"] = True
                            state["status"] = "Listening"

                            clear_audio_queue()
                            continue

                        # Disable calibration
                        elif key == "d":

                            final = segmenter.finish()

                            if final is not None:

                                uid, audio = final

                                asr_queue.put(
                                    ("final", uid, audio)
                                )

                            segmenter.energy_threshold = MIN_FRAME_RMS

                            state["threshold"] = MIN_FRAME_RMS
                            state["calibrated"] = False
                            state["speaking"] = False
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

                        if event == "started":

                            state["active_id"] = uid
                            state["speaking"] = True
                            state["partial_text"] = ""
                            state["status"] = "Speech detected"

                        elif event == "partial":

                            state["speaking"] = True

                            # Avoid building a large backlog.
                            if asr_queue.qsize() < 3:

                                asr_queue.put_nowait(
                                    ("partial", uid, audio)
                                )

                        elif event == "final":

                            state["status"] = "Finalizing"
                            state["speaking"] = False

                            asr_queue.put(
                                ("final", uid, audio)
                            )

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

                # Finish queued recognition jobs.
                asr_queue.put(None)

                worker.join()

                process_results(state, history)

                # Stop an active generation before shutting down the chat worker.
                cancel_event.set()
                chat_queue.put(None)
                chat_worker.join(timeout=3.0)
                process_chat_results(state, history)

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
            f"\n[green]Transcript saved:[/green] {OUTPUT_FILE.resolve()}"
        )


if __name__ == "__main__":
    main()


