from flask import Flask, render_template, request, send_from_directory
import os
import whisper
from gtts import gTTS
from moviepy.editor import AudioFileClip, ImageSequenceClip
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import ffmpeg
import tempfile
import re

app = Flask(__name__)

# Setup folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_VIDEO = 'static/output.mp4'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emoji map
EMOJI_MAP = {
    "money": "💰", "love": "❤️", "fire": "🔥", "cool": "😎",
    "happy": "😊", "sad": "😢", "win": "🏆", "star": "⭐",
    "success": "✅", "zodiac": "🔮", "age": "👤", "net worth": "💵"
}

def inject_emojis(text):
    for k, v in EMOJI_MAP.items():
        text = text.replace(k, f"{k} {v}")
    return text

def auto_punctuate(text):
    text = re.sub(r'([a-zA-Z0-9])(\s)([A-Z])', r'\1.\2\3', text)
    if not text.strip().endswith('.'):
        text += '.'
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        summary_text = request.form["summary"]
        images = request.files.getlist("images")
        font = ImageFont.truetype("Poppins-Bold.ttf", 80)

        # Save uploaded images
        image_paths = []
        for img in images:
            if img.filename:
                path = os.path.join(UPLOAD_FOLDER, img.filename)
                img.save(path)
                image_paths.append(path)

        if not summary_text or not image_paths:
            return "Summary text and at least one image required."

        # Step 1: Generate voice
        audio_path = "static/voice.mp3"
        clean_text = auto_punctuate(summary_text)
        gTTS(text=clean_text, lang='en').save(audio_path)
        sound = AudioSegment.from_mp3(audio_path)
        sound += AudioSegment.silent(duration=500)
        sound.export(audio_path, format="mp3")

        audio = AudioFileClip(audio_path).audio_fadeout(1)
        duration = audio.duration
        fps = 24
        total_frames = int(duration * fps)

        # Step 2: Get captions using Whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True, language="en")
        words_info = []
        for seg in result["segments"]:
            for word in seg["words"]:
                words_info.append((word["word"], word["start"], word["end"]))

        # Step 3: Create slideshow frames
        WIDTH, HEIGHT = 720, 1280
        slideshow = ImageSequenceClip(image_paths, fps=1).resize((WIDTH, HEIGHT)).set_duration(duration)
        slideshow.write_videofile("static/slideshow.mp4", fps=fps)
        cap = cv2.VideoCapture("static/slideshow.mp4")
        bg_images = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            bg_images.append(cv2.resize(frame, (WIDTH, HEIGHT)))
        cap.release()

        # Step 4: Caption overlay on frames
        out = cv2.VideoWriter("static/temp_no_audio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT))
        for i in range(total_frames):
            t = i / fps
            frame = bg_images[i]
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            for word, start, end in words_info:
                if start <= t <= end:
                    word = inject_emojis(word.strip())
                    progress = (t - start) / max((end - start), 0.001)
                    bounce = int(np.sin(progress * np.pi) * 40)
                    x, y = WIDTH // 2, HEIGHT // 2 - bounce

                    # Outline
                    for dx in [-2, 0, 2]:
                        for dy in [-2, 0, 2]:
                            draw.text((x + dx, y + dy), word, font=font, fill="black", anchor="mm")
                    draw.text((x, y), word, font=font, fill="yellow", anchor="mm")
                    break

            final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(final_frame)
        out.release()

        # Step 5: Merge audio + video
        ffmpeg.concat(
            ffmpeg.input("static/temp_no_audio.mp4").video,
            ffmpeg.input(audio_path).audio,
            v=1, a=1
        ).output(OUTPUT_VIDEO).run(overwrite_output=True)

        return render_template("index.html", video_url="/static/output.mp4")

    return render_template("index.html")
