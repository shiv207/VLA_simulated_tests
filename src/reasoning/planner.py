import os
from groq import Groq
import json

class GroqPlanner:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.system_prompt = """
        You are an advanced robot VLA (Vision-Language-Action) planner.
        You receive a human language TASK and a detailed SCENE STATE (JSON).
        
        Available Actions:
        - "move_arm_to(object_name)" -> Moves the arm's fingertips to the center of the named object.
        - "close_gripper" -> Closes the gripper. Use this when the arm is at the object.
        - "open_gripper" -> Opens the gripper.
        - "lift" -> Special vertical movement to safely lift the grasped object.
        
        Rules:
        1. Contextual Awareness: Look at the "objects" list. Identify which object matches the human's request (e.g., if asked for "red", use "red_block").
        2. Precision: Use the exact object keys from the JSON.
        3. Sequencing: Always follow the standard sequence: 
           open_gripper -> move_arm_to(obj) -> close_gripper -> lift -> move_arm_to(target) -> open_gripper.
        4. Output Format: VALID JSON ONLY. No markdown, no commentary.
        
        Output Example:
        {"plan": ["open_gripper", "move_arm_to(red_block)", "close_gripper", "lift", "move_arm_to(target_zone)", "open_gripper"]}
        """

    def plan(self, task, state):
        user_content = f"""
        TASK: {task}
        
        SCENE STATE (Current Vision):
        {json.dumps(state, indent=2)}
        """
        
        try:
            print(f"\n[PLANNER] Reasoninig about task: {task}")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content
            plan_data = json.loads(response_text)
            return plan_data.get("plan", [])
            
        except Exception as e:
            print(f"[ERROR] Groq Plan Generation Failed: {e}")
            return []
