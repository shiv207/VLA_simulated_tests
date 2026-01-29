# 🤖 VLA Simulation System

A robust Vision-Language-Action (VLA) simulation for the SO-101 robot arm using MuJoCo.

## 🌟 Features

*   **Stable Physics**: Tuned contacts and solver parameters for reliable manipulation (1ms timestep).
*   **Generative Planning**: Uses Groq (Llama 3) to convert natural language into action plans.
*   **Vision System**:
    *   **Sim Mode**: Ground-truth state detection (fast, reliable).
    *   **Gemini Mode**: Real VLM-based perception using Google Gemini Pro Vision.
*   **Clean Architecture**: Modular design separating Vision, Reasoning, and Control.

## 🚀 Quick Start

1.  **Setup Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy mujoco python-dotenv google-generativeai groq pillow
    ```

2.  **Configure Keys**
    Create a `.env` file (based on `.env.example`) with your API keys:
    ```bash
    cp .env.example .env
    # Edit .env with your GROQ_API_KEY and GEMINI_API_KEY
    ```

3.  **Run Simulation**
    ```bash
    python main.py --task "Pick up the red box"
    ```

## 🎮 Interactive Mode

Run with interactive prompts:
```bash
python main.py --interactive
```

## 📁 Project Structure

*   `main.py`: Entry point and orchestration.
*   `src/sim/`: MuJoCo physics engine wrapper and scene management.
*   `src/reasoning/`: LLM planner integration (Groq).
*   `src/control/`: Robot action executor (Mocap/IK).
*   `src/vision/`: Perception modules (Sim & Gemini).
*   `robotstudio_so101/`: XML assets and robot definitions.

## 🛠️ Troubleshooting

*   **Viewer not launching?**
    *   On macOS, `mujoco` passive viewer runs best with the `mjpython` launcher.
    *   Try: `.venv/bin/mjpython main.py`
    *   If that fails due to path spaces, the simulation will seamlessly fall back to headless mode and simply print progress.

## 📜 License
MIT License
