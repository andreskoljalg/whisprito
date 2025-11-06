#!/usr/bin/env python3
import os, sys, subprocess

# --- Auto setup virtualenv ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
PY_BIN = os.path.join(VENV_DIR, "bin", "python3")

# If not already inside venv
if not sys.executable.startswith(VENV_DIR):
    if not os.path.exists(VENV_DIR):
        print("📦 Creating virtual environment...")
        brew_python = "/opt/homebrew/bin/python3"  # Homebrew Python
        subprocess.check_call([brew_python, "-m", "venv", VENV_DIR])
        print("✅ venv created.")

        print("📦 Installing dependencies (this may take a while the first time)...")
        subprocess.check_call([PY_BIN, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([PY_BIN, "-m", "pip", "install",
                               "torch", "transformers", "colorama", "tqdm"])

    # Relaunch inside venv
    os.execv(PY_BIN, [PY_BIN] + sys.argv)

# --- Imports (safe inside venv) ---
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import json
import ssl, os

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
from colorama import init, Fore, Style
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# --- Initialize ---
init(autoreset=True)

# Model choices
available_models = {
    "1": "openai/whisper-tiny",
    "2": "openai/whisper-base",
    "3": "openai/whisper-small",
    "4": "openai/whisper-medium",
    "5": "openai/whisper-large-v3"
}

print(Fore.CYAN + "🧠 Select a Whisper model:" + Style.RESET_ALL)
for key, name in available_models.items():
    print(f"{key}: {name}")
model_choice = input(Fore.CYAN + "Enter the number of the model to use (default 4): " + Style.RESET_ALL).strip()
model_id = available_models.get(model_choice, "openai/whisper-medium")

# Device & dtype
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(Fore.CYAN + f'🔄 Loading Whisper model: {model_id}' + Style.RESET_ALL)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation='eager'
).to(device)
processor = AutoProcessor.from_pretrained(model_id)
asr = pipeline(
    'automatic-speech-recognition',
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps='word',
    dtype=dtype,
    device=device,
    generate_kwargs={"task": "transcribe"}
)

def format_time(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600*1000)
    m, rem = divmod(rem, 60*1000)
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
                'end':   current[-1]['end'],
                'text':  ' '.join(item['text'] for item in current).strip()
            })
    for w in words:
        if not current:
            current = [w]
            continue
        last_text = current[-1]['text'].rstrip()
        if last_text.endswith(SOFT_END) or last_text.endswith(STRONG_END):
            flush_current(); current = [w]; continue
        seg_text = ' '.join(item['text'] for item in current + [w]).strip()
        seg_len, seg_dur = len(seg_text), w['end'] - current[0]['start']
        if seg_len > max_chars and seg_dur >= min_duration:
            flush_current(); current = [w]
        else:
            current.append(w)
    flush_current()
    return segments

def write_srt(segments, srt_path, strip=False):
    if not segments:
        print(Fore.YELLOW + f"⚠ No segments to write for {srt_path}" + Style.RESET_ALL)
        return
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            text = seg['text']
            if strip:
                text = text.replace('.', '').replace(',', '').lower()
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{text}\n\n")

def transcribe_file(audio_path, out_dir, save_json):
    print(Fore.CYAN + f"🔊 Transcribing: {audio_path}" + Style.RESET_ALL)
    try:
        res = asr(audio_path)
        chunks = res.get('chunks') or res.get('segments') or []
        base = os.path.splitext(os.path.basename(audio_path))[0]
        if save_json:
            jp = os.path.join(out_dir, f"{base}.json")
            with open(jp, 'w', encoding='utf-8') as jf:
                import json; json.dump(chunks, jf, ensure_ascii=False, indent=2)
            print(Fore.GREEN + f"✔ JSON saved: {jp}" + Style.RESET_ALL)
        return chunks
    except Exception as e:
        print(Fore.RED + f"❌ Failed ASR on {audio_path}: {e}" + Style.RESET_ALL)
        return None

def process_file(audio_path, out_dir, max_chars, min_duration, strip_text, save_json):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = transcribe_file(audio_path, out_dir, save_json)
    if not chunks: return
    words = parse_whisper_chunks(chunks)
    if not words:
        print(Fore.YELLOW + f"⚠ No words for {base}" + Style.RESET_ALL)
        return
    segs = group_words(words, max_chars, min_duration)
    for seg in segs:
        if seg['end'] <= seg['start']:
            seg['end'] = seg['start'] + min_duration
    srt_fp = os.path.join(out_dir, f"{base}.srt")
    write_srt(segs, srt_fp, strip_text)
    print(Fore.GREEN + f"✅ SRT saved: {srt_fp}" + Style.RESET_ALL)

def main():
    try: max_c = int(input(Fore.CYAN + 'Max characters per segment: ' + Style.RESET_ALL).strip())
    except: max_c = 30
    try: min_d = float(input(Fore.CYAN + 'Min segment duration (sec): ' + Style.RESET_ALL).strip())
    except: min_d = 1.0
    strip = input(Fore.CYAN + 'Strip punctuation & lowercase? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'
    save_j = input(Fore.CYAN + 'Save raw JSON output as well? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'

    root = tk.Tk(); root.withdraw(); root.update()
    files = filedialog.askopenfilenames(
        title='🎧 Select audio files to transcribe',
        filetypes=[('Audio files','*.wav *.mp3 *.m4a'),('All files','*.*')]
    )
    if not files:
        print(Fore.RED + 'No files selected. Exiting.' + Style.RESET_ALL); return
    print(Fore.CYAN + f"\n🎧 You selected {len(files)} file(s):" + Style.RESET_ALL)
    for f in files: print(Fore.YELLOW + '- ' + os.path.basename(f) + Style.RESET_ALL)
    root.update()
    out_dir = filedialog.askdirectory(title='📁 Select output folder')
    if not out_dir:
        print(Fore.RED + 'No output folder. Exiting.' + Style.RESET_ALL); return
    root.destroy()

    with tqdm(total=len(files), desc=Fore.GREEN + '🎬 Transcribing files' + Style.RESET_ALL) as pbar:
        with ThreadPoolExecutor(max_workers=1) as ex:
            futures = {ex.submit(process_file, f, out_dir, max_c, min_d, strip, save_j): f for f in files}
            for fut in as_completed(futures):
                try: fut.result()
                except Exception as e:
                    print(Fore.RED + f"⚠️ Error on {futures[fut]}: {e}" + Style.RESET_ALL)
                finally: pbar.update(1)

if __name__ == '__main__':
    main()