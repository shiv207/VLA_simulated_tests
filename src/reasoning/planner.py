import os
from groq import Groq
import json

class GroqPlanner:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.system_prompt = """
        You are an advanced robot VLA planner.
        You receive a human language TASK and a SCENE STATE (JSON).
        
        Available Actions:
        - "move_arm_to(object_name)"
        - "close_gripper"
        - "open_gripper"
        - "lift"
        
        Rules:
        1. Contextual Awareness: Identify which object matches the request.
        2. Precision: Use exact object keys.
        3. Sequencing: open_gripper -> move_arm_to(obj) -> close_gripper -> lift -> move_arm_to(target) -> open_gripper.
        4. Output Format: VALID JSON ONLY.
        
        Output Example:
        {"plan": ["open_gripper", "move_arm_to(red_block)", "close_gripper", "lift", "move_arm_to(target_zone)", "open_gripper"]}
        """

    def plan(self, task, state):
        user_content = f"""
        TASK: {task}
        SCENE STATE:
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
