"""Deep Q-Network agent implementation without external ML dependencies."""
from __future__ import annotations

import math
import random
from typing import List

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import board_to_features


class LinearLayer:
    """Simple linear layer for small neural networks."""

    def __init__(self, input_size: int, output_size: int, rng: random.Random) -> None:
        self.input_size = input_size
        self.output_size = output_size
        bound = 1.0 / math.sqrt(input_size)
        self.weights: List[List[float]] = [
            [rng.uniform(-bound, bound) for _ in range(output_size)] for _ in range(input_size)
        ]
        self.bias: List[float] = [0.0 for _ in range(output_size)]
        self.last_input: List[float] | None = None

    def forward(self, inputs: List[float]) -> List[float]:
        self.last_input = inputs
        outputs: List[float] = []
        for j in range(self.output_size):
            total = self.bias[j]
            for i in range(self.input_size):
                total += inputs[i] * self.weights[i][j]
            outputs.append(total)
        return outputs

    def backward(self, grad_output: List[float], lr: float) -> List[float]:
        if self.last_input is None:
            raise RuntimeError("No cached input for backward pass.")
        grad_input = [0.0 for _ in range(self.input_size)]
        for i in range(self.input_size):
            for j in range(self.output_size):
                grad_input[i] += grad_output[j] * self.weights[i][j]
                self.weights[i][j] -= lr * grad_output[j] * self.last_input[i]
        for j in range(self.output_size):
            self.bias[j] -= lr * grad_output[j]
        return grad_input


def relu(values: List[float]) -> List[float]:
    return [max(0.0, v) for v in values]


def relu_grad(values: List[float], grad_output: List[float]) -> List[float]:
    return [g if v > 0 else 0.0 for v, g in zip(values, grad_output)]


class SimpleDQN:
    """Minimal fully connected network approximating Q-values."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.layer1 = LinearLayer(16, 128, rng)
        self.layer2 = LinearLayer(128, 128, rng)
        self.layer3 = LinearLayer(128, 4, rng)
        self.last_hidden1: List[float] | None = None
        self.last_hidden2: List[float] | None = None

    def forward(self, features: List[float]) -> List[float]:
        hidden1_linear = self.layer1.forward(features)
        hidden1 = relu(hidden1_linear)
        hidden2_linear = self.layer2.forward(hidden1)
        hidden2 = relu(hidden2_linear)
        output = self.layer3.forward(hidden2)
        self.last_hidden1 = hidden1_linear
        self.last_hidden2 = hidden2_linear
        return output

    def clone(self) -> "SimpleDQN":
        """Create a deep copy of the network."""
        new_network = SimpleDQN(self.rng)
        for src_layer, dest_layer in (
            (self.layer1, new_network.layer1),
            (self.layer2, new_network.layer2),
            (self.layer3, new_network.layer3),
        ):
            for i in range(len(src_layer.weights)):
                for j in range(len(src_layer.weights[i])):
                    dest_layer.weights[i][j] = src_layer.weights[i][j]
            for j in range(len(src_layer.bias)):
                dest_layer.bias[j] = src_layer.bias[j]
        return new_network

    def backward(self, grad_output: List[float], lr: float) -> None:
        if self.last_hidden2 is None or self.last_hidden1 is None:
            raise RuntimeError("No cached activations for backward pass.")
        grad_layer3 = grad_output
        grad_hidden2 = self.layer3.backward(grad_layer3, lr)
        grad_hidden2 = relu_grad(self.last_hidden2, grad_hidden2)
        grad_hidden1 = self.layer2.backward(grad_hidden2, lr)
        grad_hidden1 = relu_grad(self.last_hidden1, grad_hidden1)
        self.layer1.backward(grad_hidden1, lr)


class DQNAgent(BaseAgent):
    """Agent backed by a simple Deep Q-Network with epsilon-greedy policy."""

    def __init__(
        self,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.network = SimpleDQN(self.rng)

    def select_action(self, env: Game2048Env) -> int:
        valid_actions = env.valid_moves()
        if not valid_actions:
            return 0
        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        q_values = self.network.forward(board_to_features(env.board))
        best_action = max(valid_actions, key=lambda a: q_values[a])
        return int(best_action)

    def predict(self, features: List[float]) -> List[float]:
        """Forward pass for external callers."""
        return self.network.forward(features)

    def train_step(self, features: List[float], target_values: List[float], lr: float) -> None:
        """Single gradient step using mean squared error loss."""
        outputs = self.network.forward(features)
        grad_output = [0.0 for _ in outputs]
        for i, (out, target) in enumerate(zip(outputs, target_values)):
            grad = 2 * (out - target) / len(outputs)
            grad_output[i] = grad
        self.network.backward(grad_output, lr)


__all__ = ["DQNAgent", "SimpleDQN", "LinearLayer"]
