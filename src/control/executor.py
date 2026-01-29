import re
import numpy as np
import time

WRIST_OFFSET = 0.12
HOVER_HEIGHT = 0.10

def execute_plan(sim, get_state_fn, plan):
    print(f"Executor: Running {len(plan)} steps...")
    
    for step in plan:
        _execute_step(sim, get_state_fn, step)
        sim.wait(0.5)
    
    return True

def _execute_step(sim, get_state_fn, action):
    action = action.strip()
    
    if action == "close_gripper":
        _update_gripper(sim, "closed", wait=1.0)
        
    elif action == "open_gripper":
        _update_gripper(sim, "open", wait=0.5)
        
    elif action == "lift":
        _lift_arm(sim, get_state_fn)

    elif action.startswith("move_arm_to"):
        _move_to_object(sim, get_state_fn, action)

def _update_gripper(sim, state, wait=0.5):
    current_pos = sim.get_object_position("hand_target")
    sim.set_mocap_target(current_pos, gripper_state=state)
    sim.wait(wait)

def _lift_arm(sim, get_state_fn):
    current = sim.get_object_position("hand_target")
    target = current.copy()
    target[2] += 0.15
    
    # Get current gripper state from vision/sim
    gripper_state = get_state_fn()["robot_state"]["gripper"]
    
    sim.set_mocap_target(target, gripper_state=gripper_state)
    sim.wait(1.5)

def _move_to_object(sim, get_state_fn, action):
    target_name = _parse_target(action)
    if not target_name: return

    scene = get_state_fn()
    target_obj = scene["objects"].get(target_name)
    
    if not target_obj:
        print(f"Error: '{target_name}' not visible.")
        return

    target_pos = np.array(target_obj["position"])
    gripper_state = scene["robot_state"]["gripper"]

    hover_pos = target_pos.copy()
    hover_pos[2] += WRIST_OFFSET + HOVER_HEIGHT
    sim.set_mocap_target(hover_pos, gripper_state=gripper_state)
    sim.wait(1.5)

    final_pos = target_pos.copy()
    
    if "zone" in target_name:
        final_pos[2] += WRIST_OFFSET + 0.08
    else:
        final_pos[2] += WRIST_OFFSET

    sim.set_mocap_target(final_pos, gripper_state=gripper_state)
    sim.wait(2.0)

def _parse_target(action_str):
    match = re.search(r'\((.*?)\)', action_str)
    return match.group(1) if match else None
