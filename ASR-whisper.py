#!/usr/bin/env python3
import os, sys, shutil, subprocess

# --- Auto setup virtualenv ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
PY_BIN = os.path.join(VENV_DIR, "bin", "python3")
VENV_FLAG = "TRANSCRIBE_IN_VENV"

# If not already inside the venv (checked via env flag, not path string
# comparison, since path resolution/symlinks can make startswith() unreliable)
if os.environ.get(VENV_FLAG) != "1":
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")

        # Find a system python3 without hardcoding a Homebrew-only path
        candidates = [
            shutil.which("python3.11"),
            shutil.which("python3.12"),
            shutil.which("python3"),
            "/opt/homebrew/bin/python3",   # Apple Silicon Homebrew
            "/usr/local/bin/python3",      # Intel Homebrew
            "/usr/bin/python3",            # System python
        ]
        system_python = next((c for c in candidates if c and os.path.exists(c)), None)
        if not system_python:
            print("Could not find a python3 interpreter to create the venv with.")
            sys.exit(1)

        subprocess.check_call([system_python, "-m", "venv", VENV_DIR])
        print("venv created.")

        print("Installing dependencies (this may take a while the first time)...")
        subprocess.check_call([PY_BIN, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([PY_BIN, "-m", "pip", "install",
                               "torch", "transformers", "accelerate",
                               "colorama", "tqdm", "librosa", "soundfile"])

    # Relaunch inside venv with the flag set so we don't loop
    env = os.environ.copy()
    env[VENV_FLAG] = "1"
    os.execve(PY_BIN, [PY_BIN] + sys.argv, env)

# --- Imports (safe inside venv) ---
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import json
import ssl
import time
import threading

# Common CA bundle locations on macOS Homebrew
possible_certs = [
    "/opt/homebrew/etc/openssl@3/cert.pem",
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt"
]
for path in possible_certs:
    if os.path.exists(path):
        os.environ['SSL_CERT_FILE'] = path
        os.environ['REQUESTS_CA_BUNDLE'] = path
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=path)
        break

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import init, Fore, Style
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig, pipeline
import transformers
import warnings

# Keep library-level logging/warnings from bypassing tqdm.write and corrupting
# the progress bar. Anything printed outside our log() helper - e.g. HF's
# "Device set to use mps" info log, or the chunk_length_s experimental
# warning - writes straight to stderr without coordinating with tqdm's redraw.
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*chunk_length_s.*experimental.*")

# --- Initialize ---
init(autoreset=True)


def log(msg):
    # Use tqdm.write instead of print for anything emitted while a tqdm bar
    # is active - plain print() pushes new lines and makes the bar look
    # stuck/broken because it keeps getting redrawn below your output.
    tqdm.write(msg)


def pick_files_and_output_dir():
    """
    Run the Tkinter file/folder pickers in a short-lived, separate process.

    Why: the moment any process creates a Tk window (even withdrawn), macOS
    registers it as a GUI app for the rest of its life - that tag doesn't go
    away when the window is destroyed. If *this* process then goes on to do
    long, synchronous, blocking GPU work on the main thread, it never returns
    to Cocoa's event loop, and macOS's watchdog marks it "(Not Responding)" in
    Force Quit - even though nothing is actually wrong, it's just busy.

    By running the picker in a disposable subprocess and doing all the heavy
    transcription work in *this* plain console process (no Tkinter import at
    all here), the transcription process can never get mislabeled that way,
    no matter how long a file takes.
    """
    picker_code = r'''
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.update()
files = filedialog.askopenfilenames(
    title="Select audio files to transcribe",
    filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"), ("All files", "*.*")]
)
if not files:
    root.destroy()
    sys.exit(1)
root.update()
out_dir = filedialog.askdirectory(title="Select output folder")
root.destroy()
if not out_dir:
    sys.exit(1)
print(out_dir)
for f in files:
    print(f)
'''
    result = subprocess.run([sys.executable, "-c", picker_code], capture_output=True, text=True)
    if result.returncode != 0:
        return [], None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return [], None
    out_dir, files = lines[0], lines[1:]
    return files, out_dir


# Model choices
DEFAULT_MODEL_KEY = "2"
available_models = {
    "1": "openai/whisper-large-v3",
    "2": "TalTechNLP/whisper-large-v3-turbo-et-verbatim",  # Estonian, verbatim-tuned
}

print(Fore.CYAN + "Select a Whisper model:" + Style.RESET_ALL)
for key, name in available_models.items():
    print(f"{key}: {name}")
model_choice = input(
    Fore.CYAN + f"Enter the number of the model to use (default {DEFAULT_MODEL_KEY}: "
    f"{available_models[DEFAULT_MODEL_KEY]}): " + Style.RESET_ALL
).strip()
model_id = available_models.get(model_choice, available_models[DEFAULT_MODEL_KEY])

# --- Device & dtype ---
# Priority: CUDA GPU > Apple Silicon MPS > CPU
if torch.cuda.is_available():
    device = 'cuda:0'
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = 'mps'
    # float16 inference (not training) is generally fine on MPS and is
    # noticeably faster than float32. If you ever see garbled/repeated text,
    # switch this back to torch.float32.
    dtype = torch.float16
else:
    device = 'cpu'
    dtype = torch.float32

print(Fore.CYAN + f'Loading Whisper model: {model_id} (device={device}, dtype={dtype})' + Style.RESET_ALL)


def _load_model(dtype_, attn_impl):
    # transformers has renamed the dtype kwarg across versions (torch_dtype -> dtype).
    # Try the current name first, fall back to the older one for compatibility.
    try:
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype_,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_impl
        ).to(device)
    except TypeError:
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype_,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_impl
        ).to(device)


# 'sdpa' (PyTorch's fused scaled-dot-product-attention) is substantially
# faster than 'eager' attention, especially on MPS where eager mode issues
# many small kernel calls per layer/head. Try the fast combo first; if the
# installed torch/transformers version can't handle sdpa on this device or
# fp16 misbehaves, fall back to the slower-but-safe eager/float32 path.
try:
    model = _load_model(dtype, 'sdpa')
    print(Fore.CYAN + 'Using sdpa attention.' + Style.RESET_ALL)
except Exception as e:
    print(Fore.YELLOW + f"sdpa attention unavailable ({e}); falling back to eager." + Style.RESET_ALL)
    if device == 'mps':
        dtype = torch.float32
    model = _load_model(dtype, 'eager')

processor = AutoProcessor.from_pretrained(model_id)

# If the chosen model is an Estonian-specific fine-tune, pin the language so
# Whisper's language-detection step can't misidentify short or noisy clips.
generate_kwargs = {"task": "transcribe"}
if model_id.endswith("-et") or model_id.endswith("-et-verbatim"):
    generate_kwargs["language"] = "et"

# Word-level timestamps rely on `alignment_heads` in the generation config.
# Community fine-tunes are often trained from a base OpenAI checkpoint but
# don't carry that array over. If it's missing, borrow it from the matching
# base checkpoint (same architecture/size the fine-tune was built from).
BASE_MODEL_FOR_ALIGNMENT = {
    "TalTechNLP/whisper-large-v3-turbo-et-verbatim": "openai/whisper-large-v3-turbo",
    "TalTechNLP/whisper-medium-et": "openai/whisper-medium",
}

timestamp_granularity = 'word'
if getattr(model.generation_config, "alignment_heads", None) is None:
    base_id = BASE_MODEL_FOR_ALIGNMENT.get(model_id)
    borrowed = False
    if base_id:
        try:
            base_gen_cfg = GenerationConfig.from_pretrained(base_id)
            if getattr(base_gen_cfg, "alignment_heads", None):
                model.generation_config.alignment_heads = base_gen_cfg.alignment_heads
                borrowed = True
                print(Fore.YELLOW + f"Borrowed alignment_heads from {base_id} for word-level timestamps." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"Could not fetch alignment_heads from {base_id}: {e}" + Style.RESET_ALL)
    if not borrowed:
        timestamp_granularity = True  # segment-level timestamps instead of word-level
        print(Fore.YELLOW + "No alignment_heads available - falling back to segment-level timestamps." + Style.RESET_ALL)

pipeline_kwargs = dict(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=timestamp_granularity,
    device=device,
    generate_kwargs=generate_kwargs,
    # Long-form chunking: Whisper's native window is 30s, these let the
    # pipeline handle longer files without a manual sliding-window loop.
    chunk_length_s=30,
    stride_length_s=5,
    # Process the resulting chunks together instead of one at a time -
    # matters once a file is long enough to be split into multiple chunks.
    batch_size=8,
)
try:
    asr = pipeline('automatic-speech-recognition', dtype=dtype, **pipeline_kwargs)
except TypeError:
    asr = pipeline('automatic-speech-recognition', torch_dtype=dtype, **pipeline_kwargs)


def format_time(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


STRONG_END = ('.', '!', '?', '…')
SOFT_END = (',', ';', ':')


def parse_whisper_chunks(chunks):
    words = []
    for ch in chunks:
        txt = (ch.get('text') or '').strip()
        ts = ch.get('timestamp') or ch.get('timestamps')
        start = end = None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start, end = ts[0], ts[1]
        if txt and all(c in ''.join(STRONG_END + SOFT_END) for c in txt) and (start is None or end is None):
            if words:
                words[-1]['text'] += txt
            continue
        if txt and start is not None and end is not None:
            start, end = float(start), float(end)
            if end < start:
                end = start
            words.append({'start': start, 'end': end, 'text': txt})
    words.sort(key=lambda w: w['start'])
    return words


def group_words(words, max_chars, min_duration):
    segments, current = [], []

    def flush_current():
        if current:
            segments.append({
                'start': current[0]['start'],
                'end': current[-1]['end'],
                'text': ' '.join(item['text'] for item in current).strip()
            })

    for w in words:
        if not current:
            current = [w]
            continue
        last_text = current[-1]['text'].rstrip()
        if last_text.endswith(SOFT_END) or last_text.endswith(STRONG_END):
            flush_current()
            current = [w]
            continue
        seg_text = ' '.join(item['text'] for item in current + [w]).strip()
        seg_len, seg_dur = len(seg_text), w['end'] - current[0]['start']
        if seg_len > max_chars and seg_dur >= min_duration:
            flush_current()
            current = [w]
        else:
            current.append(w)
    flush_current()
    return segments


def write_srt(segments, srt_path, strip=False):
    if not segments:
        log(Fore.YELLOW + f"No segments to write for {srt_path}" + Style.RESET_ALL)
        return
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            text = seg['text']
            if strip:
                text = text.replace('.', '').replace(',', '').lower()
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{text}\n\n")


# Mutable at runtime: some fine-tuned checkpoints report an alignment_heads
# value in their generation config that looks present but isn't actually
# usable for word-level (DTW) timestamps. Rather than trying to detect that
# ahead of time, we catch the failure on first use and downgrade to
# segment-level timestamps for the rest of the run.
CURRENT_TIMESTAMP_MODE = timestamp_granularity


def _is_alignment_heads_error(exc):
    return "alignment_heads" in str(exc)


def _run_with_heartbeat(fn, audio_path, interval=15):
    """
    Run fn() while periodically printing a heartbeat line, so a genuinely
    long transcription is visibly distinguishable from a real hang - instead
    of total silence for however long the file takes.
    """
    stop_event = threading.Event()
    start = time.time()

    def _beat():
        while not stop_event.wait(interval):
            elapsed = int(time.time() - start)
            log(Fore.MAGENTA + f"  ... still working on {os.path.basename(audio_path)} ({elapsed}s elapsed)" + Style.RESET_ALL)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop_event.set()
        t.join(timeout=1)


def transcribe_file(audio_path, out_dir, save_json):
    global CURRENT_TIMESTAMP_MODE
    log(Fore.CYAN + f"Transcribing: {audio_path}" + Style.RESET_ALL)
    try:
        res = _run_with_heartbeat(lambda: asr(audio_path, return_timestamps=CURRENT_TIMESTAMP_MODE), audio_path)
    except ValueError as e:
        if CURRENT_TIMESTAMP_MODE == 'word' and _is_alignment_heads_error(e):
            log(Fore.YELLOW +
                "This model's alignment_heads aren't usable for word-level timestamps. "
                "Falling back to segment-level timestamps for the rest of this run." +
                Style.RESET_ALL)
            CURRENT_TIMESTAMP_MODE = True
            try:
                res = _run_with_heartbeat(lambda: asr(audio_path, return_timestamps=CURRENT_TIMESTAMP_MODE), audio_path)
            except Exception as e2:
                log(Fore.RED + f"Failed ASR on {audio_path}: {e2}" + Style.RESET_ALL)
                return None
        else:
            log(Fore.RED + f"Failed ASR on {audio_path}: {e}" + Style.RESET_ALL)
            return None
    except Exception as e:
        log(Fore.RED + f"Failed ASR on {audio_path}: {e}" + Style.RESET_ALL)
        return None

    chunks = res.get('chunks') or res.get('segments') or []
    base = os.path.splitext(os.path.basename(audio_path))[0]
    if save_json:
        jp = os.path.join(out_dir, f"{base}.json")
        with open(jp, 'w', encoding='utf-8') as jf:
            json.dump(chunks, jf, ensure_ascii=False, indent=2)
        log(Fore.GREEN + f"JSON saved: {jp}" + Style.RESET_ALL)
    return chunks


def process_file(audio_path, out_dir, max_chars, min_duration, strip_text, save_json):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = transcribe_file(audio_path, out_dir, save_json)
    if not chunks:
        return
    words = parse_whisper_chunks(chunks)
    if not words:
        log(Fore.YELLOW + f"No words for {base}" + Style.RESET_ALL)
        return
    segs = group_words(words, max_chars, min_duration)
    for seg in segs:
        if seg['end'] <= seg['start']:
            seg['end'] = seg['start'] + min_duration
    srt_fp = os.path.join(out_dir, f"{base}.srt")
    write_srt(segs, srt_fp, strip_text)
    log(Fore.GREEN + f"SRT saved: {srt_fp}" + Style.RESET_ALL)


def main():
    try:
        max_c = int(input(Fore.CYAN + 'Max characters per segment: ' + Style.RESET_ALL).strip())
    except ValueError:
        max_c = 30
    try:
        min_d = float(input(Fore.CYAN + 'Min segment duration (sec): ' + Style.RESET_ALL).strip())
    except ValueError:
        min_d = 1.0
    strip = input(Fore.CYAN + 'Strip punctuation & lowercase? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'
    save_j = input(Fore.CYAN + 'Save raw JSON output as well? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'

    files, out_dir = pick_files_and_output_dir()
    if not files:
        print(Fore.RED + 'No files selected. Exiting.' + Style.RESET_ALL)
        return
    print(Fore.CYAN + f"\nYou selected {len(files)} file(s):" + Style.RESET_ALL)
    for f in files:
        print(Fore.YELLOW + '- ' + os.path.basename(f) + Style.RESET_ALL)
    if not out_dir:
        print(Fore.RED + 'No output folder. Exiting.' + Style.RESET_ALL)
        return

    # logging_redirect_tqdm reroutes any Python `logging` module output
    # (from transformers, torch, huggingface_hub, etc.) through tqdm.write
    # for the duration of the loop, so nothing outside our own log() calls
    # can still interleave badly with the bar's redraw.
    with logging_redirect_tqdm():
        # Single worker on a single model/device - run sequentially rather than
        # spinning up a thread pool that would only ever use one thread anyway.
        for f in tqdm(files, desc=Fore.GREEN + 'Transcribing files' + Style.RESET_ALL):
            try:
                process_file(f, out_dir, max_c, min_d, strip, save_j)
            except Exception as e:
                log(Fore.RED + f"Error on {f}: {e}" + Style.RESET_ALL)


if __name__ == '__main__':
    main()