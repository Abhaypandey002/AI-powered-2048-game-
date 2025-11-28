# AI-powered 2048 Game Engine

## Project Overview
This repository contains a production-ready, modular 2048 AI system with two subsystems:

- **Internal AI Engine** (`ai_2048_engine/`): a full simulator, multiple agents (Random, Heuristic, Expectimax, DQN), training pipeline, CLI, and tests.
- **Real-Game Automation** (`real_2048_bot.py`): connects a trained agent to the actual 2048 game running in a browser/app using OCR and keyboard automation.

## Architecture
```
ai_2048_engine/
├── env/              # Game2048Env and board utilities
├── agents/           # Random, Heuristic, Expectimax, DQN agents
├── training/         # DQN trainer and evaluation helpers
├── ui/               # CLI runner for play/train/eval
└── tests/            # Pytest suite for env and agents
real_2048_bot.py      # Automation bridge to live 2048 games
requirements.txt      # Python dependencies
```

## Internal 2048 Engine
- **Environment**: `env/game_2048_env.py` exposes `reset`, `step`, `valid_moves`, `is_terminal`, and `render` using a NumPy-based 4×4 board and deterministic RNG. Helpers in `env/utils.py` handle move application, tile spawning, empty-cell discovery, and log2 feature extraction.
- **Agents**:
  - `RandomAgent`: uniform random valid move.
  - `HeuristicAgent`: uses empty-tile count, monotonicity, smoothness, and corner-max bonus.
  - `ExpectimaxAgent`: depth-limited expectimax with chance nodes for tile spawns.
  - `DQNAgent`: PyTorch feed-forward network (16→128→128→4) with epsilon-greedy policy and checkpoint save/load.
- **Training**: `training/dqn_trainer.py` implements replay buffer, epsilon decay, target network updates, and checkpointing. `training/eval.py` evaluates any agent over multiple episodes.
- **CLI**: `ui/cli_runner.py` provides commands to play, train DQN, or evaluate agents.

## Real-Game Automation Layer
`real_2048_bot.py` captures the on-screen 2048 board with `pyautogui`, extracts tiles using OpenCV and `pytesseract`, feeds the board to a `DQNAgent`, and issues keystrokes to the live game.

Key functions:
- `capture_board(region)`: screenshot of the board region.
- `extract_board_from_screenshot(image)`: split into 4×4 grid and OCR each cell via `ocr_cell`.
- `RealGameEnvAdapter`: minimal adapter exposing `board` and `valid_moves` for agent compatibility.
- Continuous loop runs OCR → agent action → `pyautogui.press` every ~200ms.

## Installing & Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## How to Play with AI
```bash
python -m ai_2048_engine.ui.cli_runner play --agent heuristic --episodes 1 --render
```
Available agents: `random`, `heuristic`, `expectimax`, `dqn`.

## How to Train DQN
```bash
python -m ai_2048_engine.ui.cli_runner train-dqn --episodes 500 --checkpoint-path checkpoints/dqn_2048.pt
```

## How to Evaluate Agents
```bash
python -m ai_2048_engine.ui.cli_runner eval --agent expectimax --episodes 20
```

## Connect AI to the Real 2048 Game
1. Ensure the browser/app with 2048 is visible on screen.
2. Adjust the `BOARD_REGION` tuple in `real_2048_bot.py` to match your board location (x, y, width, height in pixels).
3. Run:
   ```bash
   python real_2048_bot.py --checkpoint checkpoints/dqn_2048.pt
   ```
4. The script will capture the board, OCR tile values, and press movement keys until interrupted.

## Troubleshooting
- **OCR accuracy**: tweak `BOARD_REGION`, `CELL_MARGIN`, or preprocessing in `ocr_cell` if digits are misread.
- **pyautogui fails**: ensure the script has permission to capture the screen and send keystrokes.
- **Torch device**: use `--device cuda` in CLI training if a GPU is available.

## Extending the Framework
- Add agents by subclassing `BaseAgent` and implementing `select_action`.
- Swap heuristics or search strategies by editing `agents/heuristic_agent.py` or `agents/expectimax_agent.py`.
- Integrate alternative UIs by reusing `Game2048Env` and the CLI patterns.

## Testing
Run the full suite with:
```bash
python -m pytest
```
