from openai import OpenAI
from pydub import AudioSegment
import io
import os
from dotenv import load_dotenv
load_dotenv()  

client = OpenAI()

def convert_text_to_speech(text: str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=text,
    )

    # Convert the binary response content to a byte stream
    byte_stream = io.BytesIO(response.content)

    # Read the audio data from the byte stream
    audio = AudioSegment.from_file(byte_stream, format="mp3")

    # Here you can save the audio file and return its path or URL
    # so the frontend can play it
    return audio

def convert_speech_to_text(audio_file):
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )

    return transcript["text"]

if __name__ == "__main__":
    # Test the functions
    text = "What are some common libraries in Python used for data science and what are their uses?"
    audio = convert_text_to_speech(text)
    print("Audio created.")
    # Play the audio
    audio.export("output.mp3", format="mp3")
    os.system("mpg123 output.mp3")

    # # Assuming 'audio_file' is a valid audio file path
    # audio_file = "path_to_audio_file"
    # transcript = convert_speech_to_text(audio_file)
    # print("Transcript: ", transcript)
