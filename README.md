# Whisprito

Whisprito turns audio files into subtitle files. It uses Whisper (the speech-to-text model from OpenAI, run through Hugging Face Transformers) to listen to your audio and write out a `.srt` subtitle file with timestamps.

It was built with Estonian audio in mind, but it works for any language Whisper supports.

The name comes from Whisper, back when that was the only model it used. It kept the name even after a second, better model for Estonian got added.

## What it does

- Takes `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, or `.aac` files
- Writes out `.srt` subtitles, split into short chunks you control
- Can strip punctuation and lowercase everything, if you want plain text subtitles (handy for social media captions)
- Picks your GPU automatically if you have one (CUDA on Windows/Linux, Apple's MPS on Mac), and falls back to CPU if not
- Opens a normal file picker window so you can click and select your files — no typing file paths
- Can save the raw output as `.json` too, in case you want to dig into the word-by-word timing yourself

## Before you run it

You need Python 3.10 or newer installed.

That's it. The script sets up everything else by itself the first time you run it: it creates its own virtual environment in a `venv` folder next to the script, and installs the Python packages it needs (torch, transformers, and a few others). The first run will take a few minutes while it downloads these. Every run after that is fast.

If you're transcribing `.mp3` or `.m4a` files, it's a good idea to have `ffmpeg` installed on your computer, so the audio decodes correctly.

## How to run it

```bash
python3 ASR-whisper.py
```

Then just answer the questions it asks you:

1. **Which model to use** — there are two choices. One is [OpenAI's general Whisper model](https://huggingface.co/openai/whisper-large-v3). The other is [TalTechNLP's version tuned specifically for Estonian speech](https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim), and it's picked by default since it does better on Estonian audio.
2. **Max characters per subtitle line** — how long each subtitle chunk can get before it splits. Smaller numbers (like 10-15) work well for short social videos.
3. **Minimum seconds per subtitle** — how short a subtitle can be before the script tries to merge it with the next bit of speech.
4. **Strip punctuation and lowercase?** — type `y` if you want plain lowercase text with no periods or commas.
5. **Save the raw JSON too?** — type `y` if you want the word-by-word data saved alongside the subtitles, useful for debugging.

After that, a window pops up asking you to pick your audio files, then another window asking where to save the results. Pick your files, pick a folder, and the script starts working. It prints a short status update every 15 seconds so you know it hasn't frozen — long files can take a while.

When it's done, you'll have a `.srt` file (and a `.json` file, if you asked for one) for every audio file you picked, saved in the folder you chose.

## What a subtitle file looks like

```srt
1
00:00:00,000 --> 00:00:03,000
hello and welcome to our estonian ai demo

2
00:00:03,100 --> 00:00:07,000
today we're testing whisper's transcription accuracy
```

## A couple of details worth knowing

- The script tries to give you word-level timestamps (so each subtitle break lines up exactly with the words spoken). If the model you picked doesn't support that, it quietly switches to timestamps per sentence-chunk instead — your subtitles still work fine, they just group text a little differently.
- Everything runs one file at a time, in order. If one file fails for some reason, the script logs the error and moves on to the next file instead of stopping.
