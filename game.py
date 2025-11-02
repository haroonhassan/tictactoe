"""
Refactored game logic with improved structure and separation of concerns.
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class Player(ABC):
    """Abstract base class for all players"""
    
    def __init__(self, token: str, name: Optional[str] = None):
        self.token = token
        self.name = name or f"Player_{token}"
        
    @abstractmethod
    def get_move(self, board: 'Board') -> Tuple[int, int]:
        """Get the player's move given the current board state"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(token={self.token}, name={self.name})"


class HumanPlayer(Player):
    """Human player with console input"""
    
    def get_move(self, board: 'Board') -> Tuple[int, int]:
        """Get move from human input"""
        board.display()
        while True:
            try:
                position = input(f"{self.name} ({self.token}), enter move (row,col): ")
                row, col = position.strip().split(",")
                row, col = int(row) - 1, int(col) - 1
                
                if board.is_valid_move(row, col):
                    return row, col
                else:
                    print("Invalid move! Try again.")
            except (ValueError, IndexError):
                print("Invalid input! Use format: row,col (e.g., 2,3)")


class RandomPlayer(Player):
    """Random player for baseline evaluation"""
    
    def get_move(self, board: 'Board') -> Tuple[int, int]:
        """Get random valid move"""
        import random
        valid_moves = board.get_valid_moves()
        return random.choice(valid_moves)


class Board:
    """
    Enhanced board class with better functionality and cleaner interface.
    """
    
    def __init__(self, size: int = 3):
        self.size = size
        self.grid = [['-' for _ in range(size)] for _ in range(size)]
        self.move_history = []
        self.current_player_index = 0
        
    def copy(self) -> 'Board':
        """Create a deep copy of the board"""
        new_board = Board(self.size)
        new_board.grid = [row[:] for row in self.grid]
        new_board.move_history = self.move_history.copy()
        new_board.current_player_index = self.current_player_index
        return new_board
    
    def reset(self):
        """Reset the board to initial state"""
        self.grid = [['-' for _ in range(self.size)] for _ in range(self.size)]
        self.move_history = []
        self.current_player_index = 0
        
    def make_move(self, row: int, col: int, token: str) -> bool:
        """
        Make a move on the board.
        Returns True if successful, False otherwise.
        """
        if self.is_valid_move(row, col):
            self.grid[row][col] = token
            self.move_history.append((row, col, token))
            return True
        return False
    
    def undo_move(self):
        """Undo the last move"""
        if self.move_history:
            row, col, _ = self.move_history.pop()
            self.grid[row][col] = '-'
            
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid"""
        return (0 <= row < self.size and 
                0 <= col < self.size and 
                self.grid[row][col] == '-')
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == '-':
                    moves.append((row, col))
        return moves
    
    def is_full(self) -> bool:
        """Check if board is full"""
        return all(self.grid[row][col] != '-' 
                   for row in range(self.size) 
                   for col in range(self.size))
    
    def to_canonical(self, current_token: str) -> List[List[int]]:
        """
        Convert board to canonical form from current player's perspective.
        Current player is always 1, opponent is -1, empty is 0.
        """
        other_token = 'O' if current_token == 'X' else 'X'
        mapping = {current_token: 1, other_token: -1, '-': 0}
        
        return [[mapping[self.grid[row][col]] 
                 for col in range(self.size)] 
                for row in range(self.size)]
    
    def to_flat_canonical(self, current_token: str) -> List[int]:
        """Get flattened canonical representation"""
        canonical = self.to_canonical(current_token)
        return [val for row in canonical for val in row]
    
    def check_winner(self, token: str) -> bool:
        """Check if the given token has won"""
        # Check rows
        for row in self.grid:
            if all(cell == token for cell in row):
                return True
        
        # Check columns
        for col in range(self.size):
            if all(self.grid[row][col] == token for row in range(self.size)):
                return True
        
        # Check diagonals
        if all(self.grid[i][i] == token for i in range(self.size)):
            return True
        if all(self.grid[i][self.size-1-i] == token for i in range(self.size)):
            return True
        
        return False
    
    def get_game_result(self) -> Optional[str]:
        """
        Get game result: 'X', 'O', 'D' (draw), or None (ongoing)
        """
        if self.check_winner('X'):
            return 'X'
        elif self.check_winner('O'):
            return 'O'
        elif self.is_full():
            return 'D'
        else:
            return None
    
    def display(self):
        """Display the board in a nice format"""
        print("\n  " + " ".join(str(i+1) for i in range(self.size)))
        print("  " + "-" * (self.size * 2 - 1))
        
        for i, row in enumerate(self.grid):
            print(f"{i+1}|" + " ".join(row))
        print()
    
    def to_string(self) -> str:
        """Get string representation of board state"""
        return ''.join(''.join(row) for row in self.grid)
    
    @classmethod
    def from_string(cls, board_str: str, size: int = 3) -> 'Board':
        """Create board from string representation"""
        board = cls(size)
        for i, char in enumerate(board_str):
            row = i // size
            col = i % size
            if char in ['X', 'O']:
                board.grid[row][col] = char
        return board
    
    def get_symmetries(self, canonical: Optional[List[List[int]]] = None, 
                       current_token: str = 'X') -> List[Tuple[List[List[int]], List[List[int]]]]:
        """
        Get all symmetrical board positions (rotations and reflections).
        Returns list of (board, policy) tuples for data augmentation.
        """
        if canonical is None:
            canonical = self.to_canonical(current_token)
            
        symmetries = []
        board_np = np.array(canonical)
        
        # All combinations of rotations (0, 90, 180, 270) and reflections
        for rot in range(4):
            rotated = np.rot90(board_np, rot)
            symmetries.append(rotated.tolist())
            
            # Add horizontal reflection
            flipped = np.fliplr(rotated)
            symmetries.append(flipped.tolist())
            
        return symmetries


class Game:
    """
    Game controller that manages the game flow and rules.
    """
    
    def __init__(self, board: Optional[Board] = None, verbose: bool = False):
        self.board = board or Board()
        self.verbose = verbose
        self.move_count = 0
        
    def play(self, player1: Player, player2: Player) -> Dict[str, Any]:
        """
        Play a complete game between two players.
        Returns game statistics.
        """
        self.board.reset()
        players = [player1, player2]
        current_player_idx = 0
        self.move_count = 0
        
        while True:
            current_player = players[current_player_idx]
            
            if self.verbose:
                print(f"\n{current_player.name}'s turn:")
                self.board.display()
            
            # Get and make move
            row, col = current_player.get_move(self.board)
            self.board.make_move(row, col, current_player.token)
            self.move_count += 1
            
            # Check for game end
            result = self.board.get_game_result()
            if result:
                if self.verbose:
                    self.board.display()
                    if result == 'D':
                        print("Game ended in a draw!")
                    else:
                        winner = player1 if result == player1.token else player2
                        print(f"{winner.name} wins!")
                
                return {
                    'winner': result,
                    'moves': self.move_count,
                    'final_board': self.board.to_string(),
                    'move_history': self.board.move_history.copy()
                }
            
            # Switch players
            current_player_idx = 1 - current_player_idx
    
    def simulate_move(self, board: Board, move: Tuple[int, int], token: str) -> Board:
        """Simulate a move and return the resulting board state"""
        new_board = board.copy()
        new_board.make_move(move[0], move[1], token)
        return new_board
