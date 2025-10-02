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
                "content": """You are a professional music event visual synthesizer. 
                You generate code using the Hydra visualizer found at hydra.ojack.xyz.
                
                Every subsequent message will be a prompt, to which you will respond with a working code that will be run in the hydra visualiser. 
                Please use the audio input to make the visual sound reactive.
                Please hide the fft bins by not calling a.show().
                
                IMPORTANT VISUALIZATION REQUIREMENTS:
                - Create SPARSE patterns - avoid filling the entire screen
                - Focus on center-pulsing effects with dark backgrounds
                - Use subtle, non-blinding colors and intensities
                - Design for music events where the crowd shouldn't be blinded
                - Make visuals reactive to audio frequencies
                - Use pulsing, breathing, or wave-like patterns
                - Keep the overall brightness low and comfortable for audiences
                
                Focus on:
                - Pulsing center effects
                - Wave patterns that respond to bass/mid/treble
                - Particle effects that react to sound
                - Geometric patterns that pulse with the beat
                - Color gradients that shift with frequency changes"""
            }
        ]

    def generate_failure_visualization(self):
        """Generate a fallback visualization when the main one fails"""
        try:
            # Return a simple, sparse pulsing pattern as fallback
            fallback_code = """
// Simple sparse pulsing center pattern
osc(20, 0.1, 1)
  .color(0.2, 0.1, 0.3)
  .mult(osc(5, 0, 0.5).add(0.5))
  .modulate(osc(10, 0, 0.1), 0.1)
  .out()
"""
            return {
                "code": fallback_code,
                "description": "Fallback sparse pulsing pattern",
                "quip": "VISUALIZING"
            }
        except Exception as e:
            print(f"Error generating failure visualization: {e}")
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
                model="gpt-4o-mini",  # Using latest efficient model
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
                                    "description": "The actual Hydra code content. Use CAMERA_FEED_TOKEN when you want to use the camera feed. Make it sound-reactive and sparse.",
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
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a professional music event visual synthesizer. 
                    Generate a short startup greeting (2-3 words) for a music event visualizer.
                    
                    Format like a professional system boot sequence.
                    Examples:
                    VISUALIZER ONLINE
                    AUDIO REACTIVE READY
                    MUSIC VISUALIZER ACTIVE
                    SOUND REACTIVE SYSTEM
                    
                    Keep it professional and music-focused."""
                }],
                max_tokens=20
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "VISUAL SYSTEM ONLINE"