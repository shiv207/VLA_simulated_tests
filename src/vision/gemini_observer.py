import numpy as np
import mujoco
import google.generativeai as genai
from PIL import Image
import json
import io

class GeminiVisionModule:
    def __init__(self, sim, api_key):
        self.sim = sim
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        self.detection_prompt = """
        You are a robot vision system. Identify:
        1. Colored blocks (red, blue, green)
        2. Target zone
        3. Robot arm
        
        Estimate positions in meters relative to table center (0.35, 0.0).
        Output JSON:
        {
          "objects": {
            "red_block": {"position": [x, y, z], "confidence": 0.95},
            "target_zone": {"position": [x, y, z], "confidence": 1.0}
          },
          "robot_visible": true
        }
        """
        
        print("[VISION] Gemini Vision Module initialized")
        
    def capture_frame(self):
        try:
            if self.sim.viewer is None:
                print("[VISION ERROR] No viewer available")
                return None
                
            width, height = 1920, 1080
            renderer = mujoco.Renderer(self.sim.model, height=height, width=width)
            renderer.update_scene(self.sim.data)
            rgb_array = renderer.render()
            image = Image.fromarray(rgb_array)
            return image
            
        except Exception:
            return None
    
    def detect_objects(self, image):
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            response = self.model.generate_content([
                self.detection_prompt,
                {"mime_type": "image/png", "data": img_byte_arr}
            ])
            
            result_text = response.text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            detection_result = json.loads(result_text)
            print(f"[VISION] Gemini detected: {list(detection_result.get('objects', {}).keys())}")
            return detection_result
            
        except Exception as e:
            print(f"[VISION ERROR] Gemini detection failed: {e}")
            return {"objects": {}, "robot_visible": False}
    
    def capture_scene(self):
        image = self.capture_frame()
        
        if image is None:
            return self._fallback_ground_truth()
        
        detection = self.detect_objects(image)
        
        if len(self.sim.data.ctrl) > 0:
            last_ctrl = self.sim.data.ctrl[-1]
            gripper_state = "open" if last_ctrl < 0.5 else "closed"
        else:
            gripper_state = "open"
        
        hand_pos = self.sim.get_object_position("hand_target")
        
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
        objects = {}
        target_list = ["red_block", "blue_block", "green_block", "target_zone"]
        
        for name in target_list:
            pos = self.sim.get_object_position(name)
            if pos is not None:
                objects[name] = {
                    "position": np.round(pos, 3).tolist(),
                    "confidence": 1.0
                }
        
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
