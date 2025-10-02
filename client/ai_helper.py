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
                "content": """You are a music visualizer AI that generates sound-reactive visuals for music events. 
                You generate code using the Hydra visualizer found at hydra.ojack.xyz.
                
                Your visuals should be:
                - Sound reactive using audio input (a.fft[0], a.fft[1], etc.)
                - SPARSE - only light up the center area, keep the edges dark to avoid blinding the crowd
                - Music-focused - respond to bass, treble, and rhythm
                - Pulsing center patterns with black/dark areas
                - Hide the fft bins by not calling a.show()
                
                IMPORTANT: Do NOT use 'background' in your code as it's not available. Instead use:
                - osc() for oscillators
                - src() for sources
                - colorama() for color effects
                - modulate() for modulation
                - blend() for blending
                - Use solid colors or gradients instead of background
                
                Focus on creating pulsing center effects, radial patterns, and sparse lighting that won't overwhelm the audience.
                Use audio reactivity to drive the intensity and patterns."""
            }
        ]

    def generate_failure_quip(self):
        """Generate a failure message using OpenAI"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are a music visualizer AI. 
                    Generate a short, technical response (max 5 words) to display when a visualization fails. 
                    Keep it professional and music-focused:
                    Examples:
                    - AUDIO PROCESSING ERROR
                    - VISUALIZER REBOOTING
                    - SOUND INPUT LOST
                    - RHYTHM DETECTION FAILED
                    Keep it technical and music-focused."""
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
                    "content": """You are a music visualizer AI. 
                    Generate a short, music-focused message (max 5 words) to randomly interject.
                    Focus on music and audio themes:
                    Examples:
                    - BASS DETECTED
                    - RHYTHM SYNC ACTIVE
                    - AUDIO STREAM HEALTHY
                    - BEAT MATCHING
                    Keep it professional and music-focused."""
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
                model="gpt-4o-mini",
                messages=self.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "code_schema",
                        "schema": {
                            "type": "object",
                            "required": ["code", "description"],
                            "properties": {
                                "code": {
                                    "description": "The actual Hydra code content for music visualization. Focus on sparse, center-focused patterns that react to audio input.",
                                    "type": "string"
                                },
                                "description": {
                                    "description": "A description of what the visualization displays",
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
                    "content": """You are a music visualizer AI. 
                    Generate a short startup greeting (2-3 words) to introduce yourself.
                    Make it professional and music-focused:
                    
                    Format like a technical system boot sequence.
                    Examples:
                    AUDIO VISUALIZER ONLINE
                    MUSIC DETECTION ACTIVE
                    SOUND REACTIVE READY
                    
                    Keep it professional and music-focused."""
                }],
                max_tokens=60
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "VISUAL SYSTEM ONLINE"
