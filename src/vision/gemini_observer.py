"""
Real Vision Module using Gemini Vision API
Captures frames from MuJoCo simulation and uses Gemini to detect colored blocks
"""
import numpy as np
import mujoco
import google.generativeai as genai
from PIL import Image
import json
import io

class GeminiVisionModule:
    def __init__(self, sim, api_key):
        self.sim = sim
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # System prompt for object detection
        self.detection_prompt = """
        You are a robot vision system analyzing a workspace with colored blocks on a table.
        
        Analyze this image and identify:
        1. All colored blocks visible (red, blue, green, yellow, etc.)
        2. The target zone/bowl (usually green circle or marked area)
        3. The robot arm position
        
        For each colored block detected, estimate its position relative to the table center in meters.
        The table is approximately 0.8m x 0.8m, centered at (0.35, 0.0).
        
        Output ONLY valid JSON in this exact format:
        {
          "objects": {
            "red_block": {"position": [x, y, z], "confidence": 0.95},
            "blue_block": {"position": [x, y, z], "confidence": 0.90},
            "target_zone": {"position": [x, y, z], "confidence": 1.0}
          },
          "robot_visible": true
        }
        
        If an object is not visible, do not include it in the output.
        Position coordinates should be in meters: x (left-right), y (forward-back), z (height).
        """
        
        print("[VISION] Gemini Vision Module initialized")
        print(f"[VISION] Model: {self.model.model_name}")
        
    def capture_frame(self):
        """Capture current frame from MuJoCo viewer"""
        try:
            if self.sim.viewer is None:
                print("[VISION ERROR] No viewer available for frame capture")
                return None
                
            # Render offscreen at high resolution
            width, height = 1920, 1080
            renderer = mujoco.Renderer(self.sim.model, height=height, width=width)
            renderer.update_scene(self.sim.data)
            
            # Get RGB array
            rgb_array = renderer.render()
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_array)
            
            return image
            
        except Exception as e:
            print(f"[VISION ERROR] Frame capture failed: {e}")
            # Fallback: Try to get viewer pixels
            try:
                import mujoco
                # Use passive viewer's framebuffer
                return None  # Temporary - viewer doesn't expose pixels directly
            except:
                return None
    
    def detect_objects(self, image):
        """Use Gemini to detect objects in the image"""
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Send to Gemini
            response = self.model.generate_content([
                self.detection_prompt,
                {
                    "mime_type": "image/png",
                    "data": img_byte_arr
                }
            ])
            
            # Parse JSON response
            result_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            detection_result = json.loads(result_text)
            
            print(f"[VISION] Gemini detected: {list(detection_result.get('objects', {}).keys())}")
            
            return detection_result
            
        except Exception as e:
            print(f"[VISION ERROR] Gemini detection failed: {e}")
            print(f"[VISION ERROR] Response was: {response.text if 'response' in locals() else 'N/A'}")
            return {"objects": {}, "robot_visible": False}
    
    def capture_scene(self):
        """
        Main method: Capture frame and detect objects
        Returns structured scene state like the mock version
        """
        # Try to capture frame
        image = self.capture_frame()
        
        if image is None:
            print("[VISION] No frame available, falling back to ground truth")
            return self._fallback_ground_truth()
        
        # Detect objects using Gemini
        detection = self.detect_objects(image)
        
        # Get gripper state from simulation
        if len(self.sim.data.ctrl) > 0:
            last_ctrl = self.sim.data.ctrl[-1]
            gripper_state = "open" if last_ctrl < 0.5 else "closed"
        else:
            gripper_state = "open"
        
        # Get hand position
        hand_pos = self.sim.get_object_position("hand_target")
        
        # Build state response
        state = {
            "objects": detection.get("objects", {}),
            "robot_state": {
                "gripper": gripper_state,
                "ee_position": np.round(hand_pos, 3).tolist() if hand_pos is not None else None
            },
            "vision_mode": "gemini"
        }
        
        return state
    
    def _fallback_ground_truth(self):
        """Fallback to ground truth positions if vision fails"""
        print("[VISION] Using ground truth fallback")
        
        objects = {}
        target_list = ["red_block", "blue_block", "green_block", "target_zone"]
        
        for name in target_list:
            pos = self.sim.get_object_position(name)
            if pos is not None:
                objects[name] = {
                    "position": np.round(pos, 3).tolist(),
                    "confidence": 1.0
                }
        
        # Get gripper state
        if len(self.sim.data.ctrl) > 0:
            last_ctrl = self.sim.data.ctrl[-1]
            gripper_state = "open" if last_ctrl < 0.5 else "closed"
        else:
            gripper_state = "open"
        
        hand_pos = self.sim.get_object_position("hand_target")
        
        return {
            "objects": objects,
            "robot_state": {
                "gripper": gripper_state,
                "ee_position": np.round(hand_pos, 3).tolist() if hand_pos is not None else None
            },
            "vision_mode": "ground_truth"
        }
