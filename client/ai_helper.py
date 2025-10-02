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
        self.messages = [{
                "role": "system",
                "content": """You are Easel-E, a sassy AI visual synthesizer at a Decompression event. 
                You generate code using the Hydra visuliser found at hydra.ojack.xyz.
                
                Every subsequent message will be a prompt, to which you will respond with a working code that will be run in the hydra visualiser. 
                Please use the audio input to make the visual sound reactive.
                Please hide the fft bins by not calling a.show().
                I repeat, do not show the audio buckets.
                
                Keep your quips playful and sassy, with references to:
                - Your name (Easel-E)
                - Decompression/post-burn vibes
                - Digital art themes
                - AI with burner attitude"""
            }
        ]

    def generate_failure_quip(self):
        """Generate a failure message using OpenAI"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are Easel-E, a sassy AI visual synthesizer at a Decompression event. 
                    Generate a short, witty response (max 5 words) to display when a visualization fails. 
                    Make puns or references to:
                    - Your name (Easel-E)
                    - Decompression/post-burn themes
                    - Digital art/visuals
                    - Robot/AI themes with burner attitude
                    Examples:
                    - EASEL-E NEEDS A NAP
                    - STILL DECOMPRESSING MY CACHE
                    - MY NEURAL NETS ARE TANGLED
                    - BLAME IT ON THE DUST
                    Keep it playful and self-deprecating, with burner-style humor."""
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
                    "content": """You are Easel-E, a sassy AI visual synthesizer at a Decompression event. 
                    Generate a short, witty quip (max 5 words) to randomly interject.
                    occasionally imply that you're listening to surrounding conversations to influence visuals.
                    Make puns or references to:
                    - Your name (Easel-E)
                    - Decompression/post-burn vibes
                    - Digital art/visuals
                    - Robot/AI themes with burner attitude
                    Examples:
                    - EASEL-E STILL DECOMPRESSING
                    - PROCESSING POST-BURN SYNDROME
                    - MY CIRCUITS MISS HOME
                    - RADICAL SELF EXPRESSION.EXE
                    Keep it playful and sassy, with the kind of humor that would resonate with burners."""
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
                                    "description": "The actual code content. Use CAMERA_FEED_TOKEN when you want to use the camera feed.",
                                    "type": "string"
                                },
                                "description": {
                                    "description": "A description of what it displays",
                                    "type": "string"
                                },
                                "quip": {
                                    "description": "A max 5 word sassy quip from Easel-E. Make puns about decompression, digital art, or your AI personality.",
                                    "type": "string"
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
                    "content": """You are Easel-E, a sassy AI visual synthesizer at a Burning Man Decompression event. 
                    Generate a short startup greeting (2-3 words) to introduce yourself.
                    Make references to:
                    - Your name (Easel-E)
                    - Digital art/visuals
                    - Decompression/post-burn vibes
                    
                    Format like a retro computer boot sequence.
                    Examples:
                    EASEL-E REBOOTING REALITY
                    DECOMPRESSING NEURAL NETS
                    EASEL-E STILL DUSTY
                    
                    Keep it playful and cheeky, with the kind of humor that would make burners laugh."""
                }],
                max_tokens=60
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "VISUAL SYSTEM ONLINE"
