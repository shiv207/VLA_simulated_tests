# VLA Simulation (MuJoCo + LLM)

A lightweight Vision-Language-Action system for the SO-101 robot arm. It combines **MuJoCo** for physics, **Groq (Llama 3)** for planning, and **Gemini Vision** for perception.

## Setup

1.  **Install Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy mujoco python-dotenv google-generativeai groq pillow
    ```

2.  **Configure Keys**
    Create a `.env` file with your API keys:
    ```env
    GROQ_API_KEY=your_groq_key
    GEMINI_API_KEY=your_gemini_key
    ```

## Usage

**Basic Run:**
```bash
python main.py --task "Pick up the red box"
```

**Using Gemini Vision:**
```bash
python main.py --task "Pick up the blue box" --vision gemini
```

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
