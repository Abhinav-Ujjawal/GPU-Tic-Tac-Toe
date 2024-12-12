import torch
import random

def initialize_board():
    return torch.zeros((3, 3), dtype=torch.int32, device='cuda:0')

def print_board(board):
    symbols = {0: '.', 1: 'X', 2: 'O'}
    for row in board:
        print(' '.join(symbols[cell.item()] for cell in row))
    print()

def is_winner(board, player):
    for i in range(3):
        if torch.all(board[i, :] == player) or torch.all(board[:, i] == player):
            return True
    if torch.all(torch.diag(board) == player) or torch.all(torch.diag(torch.flip(board, dims=[1])) == player):
        return True
    return False

def is_draw(board):
    return not torch.any(board == 0)

def make_move_random(board, player):
    empty_positions = torch.nonzero(board == 0, as_tuple=False)
    if empty_positions.size(0) > 0:
        move = empty_positions[random.randint(0, empty_positions.size(0) - 1)]
        board[move[0], move[1]] = player

def make_move_sequential(board, player):
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = player
                return

def play_game():
    board = initialize_board()
    moves_log = []  # To keep track of moves for replay

    player_strategies = {
        1: make_move_random,  # GPU 1 uses a random strategy
        2: make_move_sequential  # GPU 2 uses a sequential strategy
    }

    current_player = 1

    while True:
        # Record the board state before the move
        moves_log.append(board.clone().cpu().numpy())
        
        # Make a move
        player_strategies[current_player](board, current_player)
        print(f"Player {current_player} makes a move:")
        print_board(board)

        # Check for a winner
        if is_winner(board, current_player):
            print(f"Player {current_player} wins!")
            moves_log.append(board.clone().cpu().numpy())
            break

        # Check for a draw
        if is_draw(board):
            print("It's a draw!")
            moves_log.append(board.clone().cpu().numpy())
            break

        # Switch player
        current_player = 3 - current_player

    return moves_log

def save_replay(moves_log, filename="replay.txt"):
    with open(filename, "w") as file:
        for i, board in enumerate(moves_log):
            file.write(f"Move {i + 1}:\n")
            for row in board:
                file.write(' '.join(map(str, row)) + "\n")
            file.write("\n")

if __name__ == "__main__":
    # Play the game
    moves_log = play_game()

    # Save the replay
    save_replay(moves_log)
    print("Replay saved to replay.txt")
