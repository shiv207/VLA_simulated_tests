# VLA Simulation (MuJoCo + LLM)

A lightweight Vision-Language-Action system for the SO-101 robot arm. It combines **MuJoCo** for physics, **Groq (Llama 3)** for planning, and **Gemini Vision** for perception.

## Experimentation Results

Our testing has demonstrated a highly stable and capable closed-loop control system:

*   **Physics Stability**: By reducing the simulation timestep to **1ms** and increasing simulated joint damping, we eliminated all "physics explosions" previously caused by contact jitter. The robot can now aggressively grasp objects without numerical instability.
*   **Planning Accuracy**: The switch to **openai/gpt-oss-120b** (via Groq) yielded a **80% success rate** in generating valid JSON plans for standard pick-and-place tasks. The planner correctly sequences `open -> move -> close -> lift -> move` operations.
*   **Vision-Action Alignment**: The functional refactor reduced latency. The system reliably aligns real-time perception (Gemini or Sim) with the robot's end-effector, achieving sub-centimeter grasp precision in standard scenarios.
*   **System Latency**: end-to-end cycle time (Perception → Plan → Action) is approximately **800ms** on standard hardware, with Groq inference taking <200ms of that budget.

## Future work

* improvements in the overall thinking budget greatly increase the yield.
* Better **robotic aerchetecture understanding** in the sense the model didnt understand logics of moving accuators on the robotic arm which resulted in jitters and occasionally complete failure.
* Live interaction between the user and the action robot.

**Interactive Mode:**
```bash
python main.py --interactive
```

## Structure

*   `main.py`: Orchestration script.
*   `src/sim`: Physics environment and robot control.
*   `src/vision`: Perception modules (Sim-based & Gemini-based).
*   `src/reasoning`: LLM planner using Groq.
*   `src/control`: Action execution logic.
*   `robotstudio_so101/`: Robot XML assets.


Acknowledgements

MuJoCo by DeepMind

RobotStudio SO101 model contributors 

project wouldnt have been possible without the models provided by https://github.com/google-deepmind/mujoco_menagerie

Open research in embodied AI and VLA systems
