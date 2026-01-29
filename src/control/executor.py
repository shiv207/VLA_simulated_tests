import re
import numpy as np

class ActionExecutor:
    WRIST_OFFSET = 0.12
    HOVER_HEIGHT = 0.10

    def __init__(self, sim, vision):
        self.sim = sim
        self.vision = vision

    def execute_plan(self, plan):
        print(f"Executor: Running {len(plan)} steps...")
        
        for step in plan:
            self._execute_step(step)
            self.sim.wait(0.5)
        
        return True

    def _execute_step(self, action):
        action = action.strip()
        
        if action == "close_gripper":
            self._update_gripper("closed", wait=1.0)
            
        elif action == "open_gripper":
            self._update_gripper("open", wait=0.5)
            
        elif action == "lift":
            self._lift_arm()

        elif action.startswith("move_arm_to"):
            self._move_to_object(action)

    def _update_gripper(self, state, wait=0.5):
        current_pos = self.sim.get_object_position("hand_target")
        self.sim.set_mocap_target(current_pos, gripper_state=state)
        self.sim.wait(wait)

    def _lift_arm(self):
        current = self.sim.get_object_position("hand_target")
        target = current.copy()
        target[2] += 0.15
        
        self.sim.set_mocap_target(target, gripper_state=self._get_current_gripper_state())
        self.sim.wait(1.5)

    def _move_to_object(self, action):
        target_name = self._parse_target(action)
        if not target_name: return

        scene = self.vision.capture_scene()
        target_obj = scene["objects"].get(target_name)
        
        if not target_obj:
            print(f"Error: '{target_name}' not visible.")
            return

        target_pos = np.array(target_obj["position"])
        gripper_state = self._get_current_gripper_state()

        hover_pos = target_pos.copy()
        hover_pos[2] += self.WRIST_OFFSET + self.HOVER_HEIGHT
        self.sim.set_mocap_target(hover_pos, gripper_state=gripper_state)
        self.sim.wait(1.5)

        final_pos = target_pos.copy()
        
        if "zone" in target_name:
            final_pos[2] += self.WRIST_OFFSET + 0.08
        else:
            final_pos[2] += self.WRIST_OFFSET

        self.sim.set_mocap_target(final_pos, gripper_state=gripper_state)
        self.sim.wait(2.0)

    def _parse_target(self, action_str):
        match = re.search(r'\((.*?)\)', action_str)
        return match.group(1) if match else None

    def _get_current_gripper_state(self):
        return self.vision.capture_scene()["robot_state"]["gripper"]
