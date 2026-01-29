import re
import numpy as np
import time

class ActionExecutor:
    def __init__(self, sim, vision):
        self.sim = sim
        self.vision = vision
        # Define offsets for safe grasping (Small Robot Scale)
        self.HOVER_HEIGHT = 0.10
        # Critical: The distance from the Mocap Control Point (Wrist) to the Fingertips
        # This prevents the robot from smashing into the table or missing the block
        self.WRIST_OFFSET_Z = 0.12 # 12cm gripper length approximate

    def execute_plan(self, plan):
        """
        Executes a list of action strings.
        Example: ["move_arm_to(red_block)", "close_gripper"]
        """
        print(f"\n[EXECUTOR] Received plan with {len(plan)} steps.")
        
        for step in plan:
            print(f"[EXECUTOR] Executing: {step}")
            self.execute_step(step)
            # Small delay to let physics settle and viewer update
            self.sim.wait(0.5) 

    def execute_step(self, action_str):
        """
        Parses a single action string and calls sim methods.
        """
        action_str = action_str.strip()
        
        if action_str == "close_gripper":
            current_pos = self.sim.get_object_position("hand_target")
            self.sim.set_mocap_target(current_pos, gripper_state="closed")
            self.sim.wait(1.0)  # Wait for grasp to engage
            return

        elif action_str == "open_gripper":
            current_pos = self.sim.get_object_position("hand_target")
            self.sim.set_mocap_target(current_pos, gripper_state="open")
            self.sim.wait(0.5)
            return
        
        elif action_str == "lift":
            current_pos = self.sim.get_object_position("hand_target")
            lift_pos = current_pos.copy()
            lift_pos[2] += 0.15 # Lift up higher to clear clutter
            self.sim.set_mocap_target(lift_pos, gripper_state=self._get_gripper_state())
            self.sim.wait(1.5)
            return

        # Handle move commands: move_arm_to(target)
        if action_str.startswith("move_arm_to"):
            target_name = self._extract_arg(action_str)
            if not target_name:
                print(f"[ERROR] Could not parse target from {action_str}")
                return
            
            # Get current scene state to find coordinates
            state = self.vision.capture_scene()
            objects = state["objects"]
            
            target_pos = None
            if target_name in objects:
                target_pos = objects[target_name]["position"]
            else:
                 print(f"[ERROR] Target '{target_name}' not seen in vision.")
                 return

            if target_pos:
                print(f"       → Target: {target_name} at {target_pos}")
                
                # 1. Hover Path: Move to (Target X, Target Y, Hover Z)
                hover_pos = list(target_pos)
                # Hover Z = Ground Z + Wrist Offset + Safety Margin
                hover_pos[2] = target_pos[2] + self.WRIST_OFFSET_Z + self.HOVER_HEIGHT
                
                print(f"       → Hover position: {[round(p, 3) for p in hover_pos]}")
                self.sim.set_mocap_target(hover_pos, gripper_state=self._get_gripper_state())
                self.sim.wait(2.0) # Wait longer for robot to reach position
                
                # 2. Descend to Grasp: Move to (Target X, Target Y, Grasp Z)
                final_pos = list(target_pos)
                
                if "zone" in target_name:
                    # Drop zone: Stay high
                    final_pos[2] += self.WRIST_OFFSET_Z + 0.10
                else:
                    # Object Grasp: Go to object Z + Wrist Offset
                    # This puts fingertips exactly at the object center
                    final_pos[2] += self.WRIST_OFFSET_Z 
                
                print(f"       → Grasp position: {[round(p, 3) for p in final_pos]}")
                self.sim.set_mocap_target(final_pos, gripper_state=self._get_gripper_state())
               
                # CRITICAL: Wait for robot to reach grasp position
                self.sim.wait(2.5)  # Longer wait to ensure position is reached
                
                # Verify position
                actual_gripper_pos = self.sim.get_object_position("gripper")
                if actual_gripper_pos is not None:
                    error = np.linalg.norm(np.array(actual_gripper_pos) - np.array(final_pos))
                    print(f"       → Position error: {error:.3f}m")

    def _extract_arg(self, action_str):
        match = re.search(r'\((.*?)\)', action_str)
        return match.group(1) if match else None

    def _get_gripper_state(self):
        state = self.vision.capture_scene()
        return state["robot_state"]["gripper"]
