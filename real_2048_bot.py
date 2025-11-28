"""Automation bridge that plays the real 2048 game using a trained agent."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image

from ai_2048_engine.agents.dqn_agent import DQNAgent
from ai_2048_engine.agents.dqn_agent import DQNAgentConfig

BOARD_REGION: Tuple[int, int, int, int] = (0, 0, 600, 600)  # x, y, width, height
CELL_MARGIN = 5
OCR_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"


class RealGameEnvAdapter:
    """Lightweight adapter exposing the board for agent compatibility."""

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((4, 4), dtype=np.int64)

    def valid_moves(self) -> list[int]:  # pragma: no cover - runtime integration
        return [0, 1, 2, 3]


def capture_board(region: Tuple[int, int, int, int]) -> Image.Image:  # pragma: no cover - requires GUI
    """Capture a screenshot of the board region."""
    return pyautogui.screenshot(region=region)


def ocr_cell(cell_image: np.ndarray) -> int:
    """Extract the numeric value from a cell image using OCR."""
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config=OCR_CONFIG).strip()
    try:
        return int(text) if text else 0
    except ValueError:
        return 0


def extract_board_from_screenshot(image: Image.Image) -> np.ndarray:
    """Split the screenshot into 4×4 grid cells and OCR each tile."""
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = image_np.shape
    cell_h, cell_w = h // 4, w // 4
    board = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            y0 = i * cell_h + CELL_MARGIN
            y1 = (i + 1) * cell_h - CELL_MARGIN
            x0 = j * cell_w + CELL_MARGIN
            x1 = (j + 1) * cell_w - CELL_MARGIN
            cell = image_np[y0:y1, x0:x1]
            board[i, j] = ocr_cell(cell)
    return board


def send_action(action: int) -> None:  # pragma: no cover - requires GUI
    """Map action index to key presses."""
    mapping = {0: "up", 1: "down", 2: "left", 3: "right"}
    pyautogui.press(mapping.get(action, "up"))


def run_loop(checkpoint_path: str, interval: float = 0.2) -> None:  # pragma: no cover - requires GUI
    """Main automation loop reading the screen, inferring an action, and pressing keys."""
    agent = DQNAgent(DQNAgentConfig(epsilon=0.0))
    agent.load_checkpoint(Path(checkpoint_path))
    adapter = RealGameEnvAdapter()
    while True:
        screenshot = capture_board(BOARD_REGION)
        adapter.board = extract_board_from_screenshot(screenshot)
        action = agent.select_action(adapter)  # type: ignore[arg-type]
        send_action(action)
        time.sleep(interval)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI parser
    parser = argparse.ArgumentParser(description="Automate the real 2048 game using a trained agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained DQN checkpoint.")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay between actions in seconds.")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=list(BOARD_REGION),
        help="Screen region for the 2048 board (x y width height).",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - entrypoint
    args = parse_args()
    global BOARD_REGION
    BOARD_REGION = tuple(args.region)  # type: ignore[assignment]
    run_loop(args.checkpoint, interval=args.interval)


if __name__ == "__main__":  # pragma: no cover - script execution
    main()
