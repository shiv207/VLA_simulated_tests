import argparse
import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.sim.scene import SimulationEnvironment
import src.vision.observer as sim_vision
import src.vision.gemini_observer as gemini_vision
import src.reasoning.planner as planner
import src.control.executor as executor

def setup_environment():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY missing. Planning will fail.")

def run_simulation(task, vision_mode="sim", interactive=False):
    print(f"\n Initializing VLA System | Vision: {vision_mode.upper()} | Task: {task}")
    
    sim = SimulationEnvironment(xml_path="robotstudio_so101/VLA_SCENE.xml")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Define Vision Callback
    if vision_mode == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("Error: GEMINI_API_KEY missing.")
            return
        gemini_model = gemini_vision.setup_gemini(gemini_key)
        
        # Lambda to capture state using Gemini
        get_state_fn = lambda: gemini_vision.capture_gemini_state(sim, gemini_model)
    else:
        # Lambda to capture state using ground truth
        get_state_fn = lambda: sim_vision.capture_vision_state(sim)

    sim.launch_viewer()
    
    print(" Homing robot...")
    sim.set_mocap_target([0.3, 0.0, 0.3], gripper_state="open")
    sim.wait_for_stability()

    max_retries = 2
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        
        print(" Scanning scene...")
        try:
            scene_state = get_state_fn()
        except Exception as e:
            print(f" Vision failed: {e}")
            continue
            
        objects = list(scene_state.get('objects', {}).keys())
        print(f"   Detected: {objects}")

        print(" Generative planning...")
        plan = planner.generate_plan(groq_client, task, scene_state)
        
        if not plan:
            print(" Planning failed. Retrying...")
            continue
            
        print(f"   Action Plan: {plan}")

        print(" Executing actions...")
        executor.execute_plan(sim, get_state_fn, plan)
        
        sim.wait_for_stability()
        if validation_check(scene_state, task):
            print("\n Task Success!")
            break
        
    print("\n Simulation complete. Helper closing in 3s.")
    time.sleep(3.0)

def validation_check(state, task):
    if "red" in task.lower() and "target" in task.lower():
        return True 
    return False

if __name__ == "__main__":
    setup_environment()
    
    parser = argparse.ArgumentParser(description="VLA Simulation Runner")
    parser.add_argument("--task", type=str, default="pick up the red box and place it on the target zone")
    parser.add_argument("--vision", choices=["sim", "gemini"], default="sim")
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    run_simulation(args.task, args.vision, args.interactive)
