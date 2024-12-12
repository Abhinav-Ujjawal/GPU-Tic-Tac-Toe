# GPU-Powered Tic-Tac-Toe Game

This project implements a simple Tic-Tac-Toe game in Python, where two GPU-based players employ different strategies to compete. The game is designed to demonstrate basic GPU computations using PyTorch.

## Features

- **Player Strategies**:
  - Player 1 (GPU 1): Makes random moves from available positions.
  - Player 2 (GPU 2): Selects the first available position sequentially.
  
- **Game Logic**:
  - Detects win conditions for rows, columns, and diagonals.
  - Ends in a draw if the board is filled without a winner.
  
- **Replay Log**:
  - The sequence of moves is recorded and saved to `replay.txt`.

- **Interactive Output**:
  - The board state is printed after each move for visualization.

## Prerequisites

- Python 3.6+
- [PyTorch](https://pytorch.org/) installed with GPU support.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
