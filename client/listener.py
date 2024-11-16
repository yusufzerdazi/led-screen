import os
import speech_recognition as sr
from concurrent import futures
import sounddevice
import json
from messager import PiMessager
from dotenv import load_dotenv
import time

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
        
        # Add conversation history
        self.max_history = 50  # Keep last 50 exchanges

        self.messages = [
            {
                "role": "system",
                "content": "You are a live code visuliser called Easel-E. You generate code using the Hydra visuliser found at hydra.ojack.xyz. Every subsequent message will be a prompt, to which you will respond with a working code that will be run in the hydra visualiser. Please use the audio input to make the visual sound reactive."
            }
        ]

        # Add accumulation variables
        self.accumulated_speech = []
        self.last_translation_time = time.time()
        self.accumulation_period = 120  # 2 minutes in seconds

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
                
                # Add retry context if needed
                if "retry" in combined_speech.lower():
                    self.messages.append({
                        "role": "system",
                        "content": "The previous visualization failed or showed a black screen. Please generate a different visualization that is more likely to work. Prefer simpler, well-tested patterns and avoid experimental features."
                    })
                
                # Add current message
                user_message = {
                    "role": "user",
                    "content": combined_speech
                }
                self.messages.append(user_message)

                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=self.messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "code_schema",
                            "schema": {
                                "type": "object",
                                "required": ["code", "description", "quip", "display"],
                                "properties": {
                                    "code": {
                                        "description": "The actual code content",
                                        "type": "string"
                                    },
                                    "description": {
                                        "description": "A description of what it displays",
                                        "type": "string"
                                    },
                                    "quip": {
                                        "description": "A max 5 word robotic quip for Easel-E to display prior to the visulizer. The target audience is attendees of Burning Man. The quip should be directed at the audience and be a bit cheeky",
                                        "type": "string"
                                    },
                                    "display": {
                                        "description": "Whether the prompt would make a cool visual or not. If a random word was picked up it should not display on screen.",
                                        "type": "boolean"
                                    }
                                },
                                "additionalProperties": False
                            }
                        }
                    }
                )

                # Update conversation history
                self.messages.append({
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                })
                
                # Trim history if too long
                if len(self.messages) > self.max_history * 2:
                    self.messages = [self.messages[0]] + self.messages[-self.max_history * 2:]

                messager.send_message(json.dumps({"type": "hydra", "content": completion.choices[0].message.content}))

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