"""
Hydra Mask Generation Mode.

Generates Hydra visuals using AI with prompts related to decompression brief.
Prompts user to approve visuals for use as internal eye masks.
Saves approved visual code to files.
"""

from .website_mode import WebsiteMode
import time
import json
import base64
import urllib.parse
import threading
import sys
import os

# Import AI helper
# Add parent directory to path to import ai_helper
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from ai_helper import AiHelper


class HydraMaskMode(WebsiteMode):
    """Mode for generating and approving Hydra visuals for eye masks"""
    
    def __init__(self, width=40, height=30):
        super().__init__(width, height)
        self.hydra_url = "http://localhost:5173"
        
        # Increase screenshot rate for better video quality
        self.screenshot_interval = 0.016  # 60 FPS (overrides parent's 25 FPS)
        
        # AI helper for generating visuals
        self.ai_helper = AiHelper()
        
        # State management
        self.current_visual_code = None
        self.current_visual_description = None
        self.waiting_for_approval = False
        self.approval_thread = None
        self.input_thread = None  # Thread for user input
        self.approval_lock = threading.Lock()
        self.user_response = None
        
        # Code saving - use absolute path relative to script location
        # __file__ is at client/modes/hydra_mask_mode.py
        # We want client/screenshots/hydra_masks
        script_dir = os.path.dirname(os.path.abspath(__file__))  # client/modes/
        client_dir = os.path.dirname(script_dir)  # client/
        self.code_output_dir = os.path.join(client_dir, "screenshots", "hydra_masks")
        
        # Ensure output directory exists
        os.makedirs(self.code_output_dir, exist_ok=True)
        print(f"Mask files will be saved to: {os.path.abspath(self.code_output_dir)}")
        
        # Generate first visual
        self.visual_generation_active = True
        
    def setup(self, **kwargs):
        """Set up Hydra URL"""
        self.url = kwargs.get('url', self.hydra_url)
    
    def init(self):
        """Initialize Hydra and start visual generation"""
        super().init()
        
        if self.driver:
            try:
                # Hide Hydra UI elements
                self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
                self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
                self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
                print("Hydra visualizer initialized")
            except Exception as e:
                print(f"Could not configure Hydra: {e}")
        
        # Start visual generation thread
        self.visual_generation_active = True
        self.generation_thread = threading.Thread(target=self._generate_and_approve_loop, daemon=True)
        self.generation_thread.start()
        
        print("Hydra Mask Mode initialized. Waiting for visual generation...")
    
    def _generate_hydra_prompt(self):
        """Generate a prompt for Hydra visual related to decompression brief"""
        decompression_prompts = [
            "Create a sparse, organic pattern suitable for an internal eye mask. Think of textures like veins, neural networks, or flowing energy patterns. Should be dark with subtle highlights.",
            "Generate a visual mask pattern inspired by decompression - flowing, organic shapes that could represent relaxation or release. Dark background with subtle glowing elements.",
            "Design a mask pattern for an eye that suggests inner vision or introspection. Organic, flowing patterns with dark areas and subtle light elements.",
            "Create a decompression-themed mask: flowing patterns that suggest release or expansion. Dark with subtle highlights, suitable for internal eye visualization.",
            "Generate an organic, flowing pattern mask inspired by decompression - think of energy flows, neural pathways, or abstract organic structures. Dark with subtle glowing elements.",
            "Design a mask pattern that suggests decompression and relaxation - flowing, organic shapes with dark areas and subtle highlights. Suitable for internal eye visualization.",
            "Create a sparse pattern mask with organic, flowing elements inspired by decompression. Dark background with subtle glowing highlights that could represent energy or flow.",
            "Generate a visual mask pattern for decompression mode - think of abstract organic structures, neural networks, or energy flows. Dark with subtle highlights.",
        ]
        
        import random
        base_prompt = random.choice(decompression_prompts)
        
        # Enhance with specific requirements
        full_prompt = f"""{base_prompt}

Requirements:
- Must be suitable as an internal mask for an eye visualization
- Should have dark areas (black/low brightness) and subtle highlights
- Organic, flowing patterns work best
- Avoid overly bright or saturated colors
- Pattern should be interesting but not distracting
- Think of textures: veins, neural networks, energy flows, organic structures

Generate Hydra code that creates this visual."""
        
        return full_prompt
    
    def _generate_visual(self):
        """Generate a new Hydra visual using AI"""
        print("\n" + "="*60)
        print("Generating new Hydra visual for decompression mask...")
        print("="*60)
        
        prompt = self._generate_hydra_prompt()
        
        try:
            # Create a temporary AI helper with decompression-specific system prompt
            temp_helper = AiHelper()
            # Override system message for decompression masks
            temp_helper.messages[0] = {
                "role": "system",
                "content": """You are a visual pattern generator AI that creates Hydra code for decompression mode eye masks.
                
                You generate code using the Hydra visualizer found at hydra.ojack.xyz.
                
                Your visuals should be:
                - Suitable as internal masks for an eye visualization
                - Dark with subtle highlights (black/low brightness areas with subtle glowing elements)
                - Organic, flowing patterns (think veins, neural networks, energy flows, abstract organic structures)
                - Sparse - not filling the entire screen
                - Interesting but not distracting
                - Avoid overly bright or saturated colors
                
                IMPORTANT: Do NOT use 'background' in your code as it's not available.
                
                Focus on creating organic, flowing patterns with dark areas and subtle highlights that work as internal eye masks."""
            }
            
            # Override model to use GPT-4o (best available model, GPT-5 doesn't exist yet)
            # We need to modify the generate_visualization method call
            # Let's call OpenAI directly with the better model
            try:
                from openai import OpenAI
                from dotenv import load_dotenv
                import os
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
                openai_client = OpenAI(api_key=api_key)
                
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": """You are a visual pattern generator AI that creates Hydra code for decompression mode eye masks.
                        
                        You generate code using the Hydra visualizer found at hydra.ojack.xyz.
                        
                        Your visuals should be:
                        - Suitable as internal masks for an eye visualization
                        - Dark with subtle highlights (black/low brightness areas with subtle glowing elements)
                        - Organic, flowing patterns (think veins, neural networks, energy flows, abstract organic structures)
                        - Sparse - not filling the entire screen
                        - Interesting but not distracting
                        - Avoid overly bright or saturated colors
                        
                        IMPORTANT: Do NOT use 'background' in your code as it's not available.
                        
                        Focus on creating organic, flowing patterns with dark areas and subtle highlights that work as internal eye masks."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # Call OpenAI with GPT-5 model
                completion = openai_client.chat.completions.create(
                    model="gpt-5",  # Using GPT-5
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "code_schema",
                            "schema": {
                                "type": "object",
                                "required": ["code", "description"],
                                "properties": {
                                    "code": {
                                        "description": "The actual Hydra code content for decompression mask visualization. Focus on organic, flowing patterns with dark areas and subtle highlights.",
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
                
                response = completion.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI directly: {e}, falling back to AiHelper")
                response = temp_helper.generate_visualization(prompt)
            
            if response:
                # Parse JSON response
                try:
                    response_data = json.loads(response)
                    code = response_data.get('code', '')
                    description = response_data.get('description', 'Unknown visual')
                    
                    self.current_visual_code = code
                    self.current_visual_description = description
                    
                    print(f"\nGenerated visual: {description}")
                    print("\n" + "="*60)
                    print("FULL HYDRA CODE:")
                    print("="*60)
                    print(code)
                    print("="*60 + "\n")
                    
                    return True
                except json.JSONDecodeError:
                    print("Error: Failed to parse AI response as JSON")
                    print(f"Response: {response[:200]}...")
                    return False
            else:
                print("Error: AI returned no response")
                return False
                
        except Exception as e:
            print(f"Error generating visual: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_visual_in_hydra(self):
        """Load the current visual code into Hydra"""
        if not self.current_visual_code or not self.driver:
            return False
        
        try:
            # Encode code and load in Hydra
            encoded_code = base64.b64encode(self.current_visual_code.encode('utf-8')).decode('utf-8')
            url = f"{self.hydra_url}?code={urllib.parse.quote_plus(encoded_code)}"
            
            self.driver.get(url)
            time.sleep(2)  # Wait for Hydra to load
            
            # Hide UI elements
            self.driver.execute_script("document.getElementById('modal').style.display = 'none';")
            self.driver.execute_script("document.getElementById('editor-container').style.display = 'none';")
            self.driver.execute_script("document.getElementById('info-container').style.display = 'none';")
            
            # Execute the code - escape properly for JavaScript
            # Replace backticks and escape quotes
            escaped_code = self.current_visual_code.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
            
            self.driver.execute_script(f"""
                if (typeof hydraSynth !== 'undefined' && hydraSynth) {{
                    try {{
                        eval(`{escaped_code}`);
                    }} catch(e) {{
                        console.log('Code execution error:', e);
                    }}
                }}
            """)
            
            print("Visual loaded in Hydra")
            return True
            
        except Exception as e:
            print(f"Error loading visual in Hydra: {e}")
            return False
    
    def _prompt_user_approval(self):
        """Prompt user for approval in console"""
        print("\n" + "-"*60)
        print(f"Visual Description: {self.current_visual_description}")
        print("-"*60)
        print("\nIs this visual suitable for use as an internal eye mask?")
        print("Type 'Y' to approve and save the code")
        print("Type 'N' to generate a new visual")
        print("Waiting for your response...")
        print("> ", end='', flush=True)  # Print prompt and flush
        
        self.waiting_for_approval = True
        self.user_response = None
        
        # Check if stdin is available and is a TTY
        import sys
        if sys.stdin.isatty():
            # stdin is a terminal - use background thread
            input_thread = threading.Thread(target=self._get_user_input, daemon=False)
            input_thread.start()
            self.input_thread = input_thread
        else:
            # stdin is not a terminal (redirected or background process)
            # Try to read from stdin anyway, but warn user
            print("\nWARNING: stdin doesn't appear to be a terminal.")
            print("If running via script, ensure stdin is connected.")
            print("Attempting to read input anyway...")
            input_thread = threading.Thread(target=self._get_user_input, daemon=False)
            input_thread.start()
            self.input_thread = input_thread
        
        # Wait for response (with timeout check in main loop)
        return True
    
    def _get_user_input(self):
        """Get user input in a separate thread"""
        import sys
        
        try:
            # Flush stdout first to ensure prompt is visible
            sys.stdout.flush()
            sys.stderr.flush()  # Also flush stderr
            
            # Debug: Check stdin status
            if not sys.stdin.isatty():
                print("\n[DEBUG] stdin is not a TTY - may have issues reading input")
                print("[DEBUG] If running via script, try running without '&' or ensure stdin is connected")
            
            # Try to read from stdin
            # Use readline() which blocks properly
            try:
                # Make sure stdin is not closed
                if sys.stdin.closed:
                    print("\nERROR: stdin is closed. Cannot read input.")
                    with self.approval_lock:
                        self.user_response = 'N'
                        self.waiting_for_approval = False
                    return
                
                # Read a line (this will block until input is available)
                line = sys.stdin.readline()
                
                if not line:
                    # EOF
                    print("\nEOF reached on stdin. Defaulting to reject.")
                    with self.approval_lock:
                        self.user_response = 'N'
                        self.waiting_for_approval = False
                    return
                
                response = line.strip().upper()
                
            except (EOFError, KeyboardInterrupt, OSError) as e:
                # Handle Ctrl+C, EOF, or I/O errors
                print(f"\nInput error: {e}. Defaulting to reject.")
                with self.approval_lock:
                    self.user_response = 'N'
                    self.waiting_for_approval = False
                return
            
            # Only accept Y or N
            if response in ['Y', 'N']:
                with self.approval_lock:
                    self.user_response = response
                    self.waiting_for_approval = False
                    print(f"\nReceived: {response}")
            elif response:
                # Invalid but non-empty response
                print(f"\nInvalid response '{response}'. Please type Y or N.")
                print("> ", end='', flush=True)
                # Try again (recursive, but only for invalid input, not empty)
                self._get_user_input()
            else:
                # Empty response - wait and try again
                print("\nEmpty response. Please type Y or N.")
                print("> ", end='', flush=True)
                self._get_user_input()
                    
        except Exception as e:
            print(f"\nError getting user input: {e}")
            import traceback
            traceback.print_exc()
            with self.approval_lock:
                self.user_response = None
                self.waiting_for_approval = False
    
    def _check_user_response(self):
        """Check if user has responded"""
        with self.approval_lock:
            if self.user_response is not None:
                response = self.user_response
                self.user_response = None
                return response
            return None
    
    def _save_code(self):
        """Save the current visual code to a file"""
        if not self.current_visual_code:
            print("Error: No code to save")
            return False
        
        try:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hydra_mask_{timestamp}.js"
            filepath = os.path.join(self.code_output_dir, filename)
            
            # Save code to file
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write description as comment
                f.write(f"// {self.current_visual_description}\n")
                f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("//\n")
                f.write("// Hydra code for decompression mode eye mask\n")
                f.write("//\n\n")
                f.write(self.current_visual_code)
            
            # Verify file was actually written
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"\n{'='*60}")
                print(f"Code saved: {filename}")
                print(f"Location: {os.path.abspath(filepath)}")
                print(f"File size: {file_size} bytes")
                print(f"{'='*60}\n")
            else:
                print(f"\nERROR: File was not created at {filepath}")
                print(f"Attempted to write to: {os.path.abspath(filepath)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error saving code: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_and_approve_loop(self):
        """Main loop for generating and approving visuals"""
        while self.visual_generation_active:
            try:
                # Generate new visual
                if not self._generate_visual():
                    print("Failed to generate visual, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Load visual in Hydra
                if not self._load_visual_in_hydra():
                    print("Failed to load visual in Hydra, retrying...")
                    time.sleep(2)
                    continue
                
                # Wait a moment for visual to render
                time.sleep(3)
                
                # Prompt user for approval
                self._prompt_user_approval()
                
                # Wait for user response (check periodically)
                timeout = 30  # 30 second timeout
                start_wait = time.time()
                
                while self.waiting_for_approval:
                    response = self._check_user_response()
                    
                    if response == 'Y':
                        # User approved - save code
                        print("\n✓ Visual approved! Saving code...")
                        if self._save_code():
                            print("Code saved successfully. Generating next visual...")
                        else:
                            print("Error saving code, but continuing to next visual...")
                        time.sleep(1)
                        break
                        
                    elif response == 'N':
                        # User rejected - generate new one
                        print("\n✗ Visual rejected. Generating new visual...")
                        time.sleep(2)
                        break
                    
                    # Check timeout
                    if time.time() - start_wait > timeout:
                        print("\nTimeout waiting for response. Generating new visual...")
                        break
                    
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in generation loop: {e}")
                time.sleep(5)
    
    def update(self):
        """Update and return current frame"""
        # Get frame from parent class
        frame = super().update()
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Hydra Mask Mode...")
        
        # Stop visual generation
        self.visual_generation_active = False
        
        # Wait for threads
        if hasattr(self, 'generation_thread'):
            self.generation_thread.join(timeout=2.0)
        
        # Clean up parent
        super().cleanup()
        
        print("Hydra Mask Mode cleaned up")

