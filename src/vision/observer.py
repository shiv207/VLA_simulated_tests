from src.sim.scene import SimulationEnvironment
import numpy as np

class VisionModule:
    def __init__(self, sim: SimulationEnvironment):
        self.sim = sim

    def capture_scene(self):
        objects = {}
        target_list = ["red_block", "blue_block", "green_block", "target_zone"]
        
        for name in target_list:
            pos = self.sim.get_object_position(name)
            if pos is not None:
                objects[name] = {
                    "position": np.round(pos, 3).tolist()
                }
        
        hand_pos = self.sim.get_object_position("hand_target") 
        
        if len(self.sim.data.ctrl) > 0:
            last_ctrl = self.sim.data.ctrl[-1]
            gripper_state = "open" if last_ctrl < 0.5 else "closed"
        else:
            gripper_state = "open"

        state = {
            "objects": objects,
            "robot_state": {
                "gripper": gripper_state,
                "ee_position": np.round(hand_pos, 3).tolist() if hand_pos is not None else None
            }
        }
        
        return state
