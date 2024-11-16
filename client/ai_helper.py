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
                    "content": "You are Easel-E, a visual synthesizer. Generate a short, witty response (max 5 words) to display when a visualization fails. The tone should be playful and self-deprecating, suitable for Burning Man attendees."
                }],
                max_tokens=20
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating failure quip: {e}")
            return "Oops, my bad!"

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

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None 