# Whisprito 🌀

**Whisprito** is a local subtitle generator powered by [Whisper](https://huggingface.co/openai/whisper-large-v3) via Hugging Face Transformers.  
It transcribes audio files into clean `.srt` subtitle files with intelligent segmentation and optional formatting.  
Perfect for podcasts, interviews, or Estonian audio content. 🇪🇪

---

## 🚀 Features

- 🎧 Transcribe `.wav`, `.mp3`, and `.m4a` audio files
- ⏱️ Smart segmenting with customizable timing and length
- ✂️ Option to strip punctuation and lowercase text
- 🧠 Multi-threaded for speed
- 🖱️ GUI file picker built with `tkinter`

---

## 📦 Requirements

- Python 3.10+
- Install dependencies:
  ```bash
  pip install torch transformers colorama tqdm
  ```

> 💡 Make sure `ffmpeg` is installed and in your system path (for audio decoding).

---


## 🛠️ Usage

1. Run the script:
   ```bash
   python3 ASR-whisper.py
   ```

2. Follow the interactive prompts:
   - Max characters per subtitle (recommended for social media: 10)
   - Min duration per segment (recommended for social media: 1)
   - Punctuation stripping 
   - Raw json saving (for debugging only)
   - Select audio files
   - Choose output folder

3. Resulting `.srt` and `.json` files will be saved in your selected folder.

---

## 🔤 Example Output

```srt
1
00:00:00,000 --> 00:00:03,000
hello and welcome to our estonian ai demo

2
00:00:03,100 --> 00:00:07,000
today we're testing whisper's transcription accuracy
```

---

## 🧠 Under the hood

- Uses Hugging Face’s 🤗 Transformers `pipeline()` API
- Model: `openai/whisper-large-v3`
- Automatically selects GPU (if available)

---
