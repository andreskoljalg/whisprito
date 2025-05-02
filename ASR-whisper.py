#!/usr/bin/env python3
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import json
import torch
from tqdm import tqdm
from colorama import init, Fore, Style
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Initialize colorama
init(autoreset=True)

# Device and dtype configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = 'openai/whisper-large-v3'

print(Fore.CYAN + '🔄 Loading Whisper model...' + Style.RESET_ALL)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
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
    torch_dtype=dtype,
    device=device
)

def format_time(seconds):
    total_millis = int(round(seconds * 1000))
    hours, rem = divmod(total_millis, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_whisper_chunks(chunks):
    words = []
    seen = set()
    for chunk in chunks:
        ts = chunk.get('timestamp') or chunk.get('timestamps')
        txt = chunk.get('text', '').strip()
        if not ts or not txt or not isinstance(ts, (list, tuple)) or len(ts) < 2:
            continue
        start, end = float(ts[0]), float(ts[1])
        if start == end:
            continue  # Skip zero-duration segments
        if end < start:
            end = start
        key = (start, end, txt)
        if key in seen:
            continue
        seen.add(key)
        words.append({'start': start, 'end': end, 'text': txt})
    words.sort(key=lambda w: w['start'])
    return words

def group_words(words, max_chars, min_duration):
    segments = []
    current = []
    for w in words:
        if not current:
            current = [w]
            continue
        seg_text = ' '.join(item['text'] for item in current + [w]).strip()
        seg_len = len(seg_text)
        seg_dur = w['end'] - current[0]['start']
        last_text = current[-1]['text']
        punct = bool(last_text and last_text[-1] in '.!?')
        if (seg_len > max_chars and seg_dur >= min_duration) or (punct and (seg_dur >= min_duration or len(current) >= 3)):
            segments.append({
                'start': current[0]['start'],
                'end': current[-1]['end'],
                'text': ' '.join(item['text'] for item in current)
            })
            current = [w]
        else:
            current.append(w)
    if current:
        segments.append({
            'start': current[0]['start'],
            'end': current[-1]['end'],
            'text': ' '.join(item['text'] for item in current)
        })
    return segments

def write_srt(segments, srt_path, strip=False):
    if not segments:
        print(Fore.YELLOW + f"⚠ No segments to write for {srt_path}" + Style.RESET_ALL)
        return
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, start=1):
            text = seg['text']
            if strip:
                text = text.replace('.', '').replace(',', '').lower()
            start_ts = format_time(seg['start'])
            end_ts = format_time(seg['end'])
            f.write(f'{idx}\n{start_ts} --> {end_ts}\n{text}\n\n')

def transcribe_file(audio_path, out_dir, save_json):
    print(Fore.CYAN + f"🔊 Transcribing: {audio_path}" + Style.RESET_ALL)
    try:
        res = asr(audio_path)
        chunks = res.get('chunks') or res.get('segments') or []
        base = os.path.splitext(os.path.basename(audio_path))[0]
        if save_json:
            json_fp = os.path.join(out_dir, f"{base}.json")
            with open(json_fp, 'w', encoding='utf-8') as jf:
                json.dump(chunks, jf, ensure_ascii=False, indent=2)
            print(Fore.GREEN + f"✔ JSON saved: {json_fp}" + Style.RESET_ALL)
        return chunks
    except Exception as e:
        print(Fore.RED + f"❌ Failed ASR on {audio_path}: {e}" + Style.RESET_ALL)
        return None

def process_file(audio_path, out_dir, max_chars, min_duration, strip_text, save_json):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = transcribe_file(audio_path, out_dir, save_json)
    if not chunks:
        return
    words = parse_whisper_chunks(chunks)
    if not words:
        print(Fore.YELLOW + f"⚠ No words for {base}" + Style.RESET_ALL)
        return
    segments = group_words(words, max_chars, min_duration)
    srt_fp = os.path.join(out_dir, f"{base}.srt")
    write_srt(segments, srt_fp, strip_text)
    print(Fore.GREEN + f"✅ SRT saved: {srt_fp}" + Style.RESET_ALL)

def main():
    try:
        max_chars = int(input(Fore.CYAN + 'Max characters per segment: ' + Style.RESET_ALL).strip())
    except:
        max_chars = 30
    try:
        min_duration = float(input(Fore.CYAN + 'Min segment duration (sec): ' + Style.RESET_ALL).strip())
    except:
        min_duration = 1.0
    strip_text = input(Fore.CYAN + 'Strip punctuation & lowercase? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'
    save_json = input(Fore.CYAN + 'Save raw JSON output as well? (y/n): ' + Style.RESET_ALL).strip().lower() == 'y'

    root = tk.Tk()
    root.withdraw()
    root.update()
    files = filedialog.askopenfilenames(
        title='🎧 Select audio files to transcribe',
        filetypes=[('Audio files', '*.wav *.mp3 *.m4a'), ('All files', '*.*')]
    )
    if not files:
        print(Fore.RED + 'No files selected. Exiting.' + Style.RESET_ALL)
        return
    print(Fore.CYAN + f"\n🎧 You selected {len(files)} file(s):" + Style.RESET_ALL)
    for f in files:
        print(Fore.YELLOW + '- ' + os.path.basename(f) + Style.RESET_ALL)
    root.update()
    out_dir = filedialog.askdirectory(title='📁 Select folder where .srt files will be saved')
    if not out_dir:
        print(Fore.RED + 'No output folder selected. Exiting.' + Style.RESET_ALL)
        return
    root.destroy()

    with tqdm(total=len(files), desc=Fore.GREEN + '🎬 Transcribing files' + Style.RESET_ALL) as pbar:
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
            futures = {
                executor.submit(
                    process_file, f, out_dir, max_chars, min_duration, strip_text, save_json
                ): f for f in files
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(Fore.RED + f"⚠️ Error on {futures[future]}: {e}" + Style.RESET_ALL)
                finally:
                    pbar.update(1)

if __name__ == '__main__':
    main()
