import numpy as np
import mujoco
import google.generativeai as genai
from PIL import Image
import json
import io

DETECTION_PROMPT = """
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

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp')

def capture_gemini_state(sim, model):
    image = _capture_frame(sim)
    
    if image is None:
        return _fallback_ground_truth(sim)
    
    detection = _detect_objects(model, image)
    
    if len(sim.data.ctrl) > 0:
        last_ctrl = sim.data.ctrl[-1]
        gripper_state = "open" if last_ctrl < 0.5 else "closed"
    else:
        gripper_state = "open"
    
    hand_pos = sim.get_object_position("hand_target")
    
    state = {
        "objects": detection.get("objects", {}),
        "robot_state": {
            "gripper": gripper_state,
            "ee_position": np.round(hand_pos, 3).tolist() if hand_pos is not None else None
        },
        "vision_mode": "gemini"
    }
    
    return state

def _capture_frame(sim):
    try:
        if sim.viewer is None:
            print("[VISION ERROR] No viewer available")
            return None
            
        width, height = 1920, 1080
        renderer = mujoco.Renderer(sim.model, height=height, width=width)
        renderer.update_scene(sim.data)
        rgb_array = renderer.render()
        image = Image.fromarray(rgb_array)
        return image
        
    except Exception:
        return None

def _detect_objects(model, image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        response = model.generate_content([
            DETECTION_PROMPT,
            {"mime_type": "image/png", "data": img_byte_arr}
        ])
        
        result_text = response.text.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        print(f"[VISION ERROR] Gemini detection failed: {e}")
        return {"objects": {}, "robot_visible": False}

def _fallback_ground_truth(sim):
    objects = {}
    target_list = ["red_block", "blue_block", "green_block", "target_zone"]
    
    for name in target_list:
        pos = sim.get_object_position(name)
        if pos is not None:
            objects[name] = {
                "position": np.round(pos, 3).tolist(),
                "confidence": 1.0
            }
    
    if len(sim.data.ctrl) > 0:
        last_ctrl = sim.data.ctrl[-1]
        gripper_state = "open" if last_ctrl < 0.5 else "closed"
    else:
        gripper_state = "open"
    
    hand_pos = sim.get_object_position("hand_target")
    
    return {
        "objects": objects,
        "robot_state": {
            "gripper": gripper_state,
            "ee_position": np.round(hand_pos, 3).tolist() if hand_pos is not None else None
        },
        "vision_mode": "ground_truth"
    }
