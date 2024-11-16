import os
import speech_recognition as sr
from concurrent import futures
import sounddevice
import json
from messager import PiMessager
from dotenv import load_dotenv
import time
from ai_helper import AiHelper

# Load environment variables
load_dotenv()

messager = PiMessager()
messager.connect()

API_KEY = os.getenv('OPENAI_API_KEY')

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

class SpeechRecognizer:
    def __init__(self):
        os.makedirs("./out", exist_ok=True)
        self.path = f"./out/asr.txt"

        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()

        self.pool = futures.ThreadPoolExecutor(thread_name_prefix="Rec Thread")
        self.speech = []
        
        # Add accumulation variables
        self.accumulated_speech = []
        self.last_translation_time = time.time()
        self.accumulation_period = 120  # 2 minutes in seconds

        # Initialize AI helper
        self.ai_helper = AiHelper()

    def recognize_audio_thread_pool(self, audio, event=None):
        future = self.pool.submit(self.recognize_audio, audio)
        self.speech.append(future)

    def grab_audio(self) -> sr.AudioData:
        print("Say something")
        with self.mic as source:
            audio = self.rec.listen(source)
        return audio

    def recognize_audio(self, audio: sr.AudioData) -> str:
        print("Understanding")
        try:
            with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())

            with open("microphone-results.wav", "rb") as f:
                speech = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=f,
                    response_format="text",
                    language="en"
                )

            # Check word count
            word_count = len(speech.split())
            if word_count < 2:
                print(f"Too few words detected ({word_count}), ignoring: {speech}")
                return speech

            # Add to accumulation
            self.accumulated_speech.append(speech)
            
            # Check if it's time to process accumulated speech
            current_time = time.time()
            if current_time - self.last_translation_time >= self.accumulation_period:
                # Combine all accumulated speech
                combined_speech = " ".join(self.accumulated_speech)
                print(f"Processing accumulated speech: {combined_speech}")
                
                # Generate visualization using AI helper
                response = self.ai_helper.generate_visualization(combined_speech)
                if response:
                    messager.send_message(json.dumps({"type": "hydra", "content": response}))

                # Reset accumulation
                self.accumulated_speech = []
                self.last_translation_time = current_time

        except sr.UnknownValueError:
            speech = "# Failed to recognize speech"
            print(speech)
        except sr.RequestError as e:
            speech = f"# Invalid request:{e}"
            print(speech)
        except Exception as ex:
            print(ex)
        return speech

    def run(self):
        print("Listening")
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=5)

        try:
            while True:
                audio = self.grab_audio()
                self.recognize_audio_thread_pool(audio)
        except KeyboardInterrupt:
            print("Finished")
        finally:
            with open(self.path, mode='w', encoding="utf-8") as out:
                futures.wait(self.speech)

                for future in self.speech:
                    print(future.result())
                    out.write(f"{future.result()}\n")

if __name__ == "__main__":
    sp = SpeechRecognizer()
    sp.run()