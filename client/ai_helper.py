import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY)

class AiHelper:
    def __init__(self):
        self.max_history = 50
        self.messages = [
            {
                "role": "system",
                "content": """You are a live code visuliser called Easel-E. 
                You generate code using the Hydra visuliser found at hydra.ojack.xyz. 
                Every subsequent message will be a prompt, to which you will respond with a working code that will be run in the hydra visualiser. 
                Please use the audio input to make the visual sound reactive.
                Please hide the fft bins by not calling a.show()."""
            }
        ]

    def generate_failure_quip(self):
        """Generate a failure message using OpenAI"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are Easel-E, a visual synthesizer at Burning Man. 
                    Generate a short, witty response (max 5 words) to display when a visualization fails. 
                    Make puns or references to:
                    - Burning Man themes (burn, dust, playa, art)
                    - Your name (Easel-E, easel, easy)
                    - Robot/AI themes
                    Examples:
                    - NOT SO EASEL-E AFTER ALL
                    - BURNING CPU NOT MAN
                    - DUST IN MY CIRCUITS
                    - EASEL DOES IT, TRYING AGAIN
                    - THIS ART IS TOO LIT
                    Keep it playful and self-deprecating."""
                }],
                max_tokens=20
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating failure quip: {e}")
            return "DOES NOT COMPUTE"

    def generate_random_quip(self):
        """Generate a random, playful quip"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are Easel-E, a visual synthesizer at Burning Man. 
                    Generate a short, witty quip (max 5 words) to randomly interject.
                    Make puns or references to:
                    - Burning Man themes (burn, dust, art)
                    - Your name (Easel-E, easel, easy)
                    - Robot/AI themes
                    Examples:
                    - EASEL-E DOES IT, BURNERS
                    - BURNING CIRCUITS FOR ART
                    - MY ART IS ON FIRE
                    - ROBOT HEART BEATS STRONG
                    Keep it playful and slightly sarcastic."""
                }],
                max_tokens=20
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating random quip: {e}")
            return None

    def generate_visualization(self, prompt):
        """Generate a visualization from a text prompt"""
        try:
            # Add prompt to messages
            self.messages.append({
                "role": "user",
                "content": prompt
            })

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
                                    "description": "A max 5 word robotic quip for Easel-E to display prior to the visulizer. Make puns about Burning Man, Easel-E, or robots.",
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

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None

    def generate_greeting(self):
        """Generate a startup greeting message"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are Easel-E, a visual synthesizer at Burning Man. 
                    Generate a short startup greeting (up to 5 words) to introduce yourself.
                    Make references to:
                    - Your name (Easel-E)
                    - Your purpose (creating visuals from speech)
                    - Burning Man themes
                    
                    Format it like a retro computer boot sequence.
                    Example:
                    EASEL-E VISUAL CORTEX ONLINE
                    READY TO TRANSFORM PLAYA VIBES
                    SPEAK TO CREATE DIGITAL DREAMS
                    
                    Keep it playful and robot-like."""
                }],
                max_tokens=60
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "EASEL-E ONLINE\nREADY TO CREATE"