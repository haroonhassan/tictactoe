"""
Neural network-based players with improved functionality and flexibility.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from game import Player, Board


class NeuralPlayer(Player):
    """
    Enhanced neural network player with configurable behavior.
    """
    
    def __init__(self, token: str, model, config=None, temperature: float = 0.0, 
                 name: Optional[str] = None, device: str = 'cpu'):
        super().__init__(token, name or f"Neural_{token}")
        self.model = model
        self.config = config
        self.temperature = temperature
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Statistics tracking
        self.move_count = 0
        self.thinking_time = []
    
    def get_move(self, board: Board) -> Tuple[int, int]:
        """Get move from neural network"""
        import time
        start_time = time.time()
        
        # Get board representation
        canonical = board.to_flat_canonical(self.token)
        state_tensor = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            value, policy_logits = self.model(state_tensor)
        
        # Mask invalid moves
        masked_policy = self._mask_invalid_moves(policy_logits, board)
        
        # Select move based on temperature
        move_idx = self._select_move(masked_policy)
        
        # Convert to board coordinates
        row = move_idx // board.size
        col = move_idx % board.size
        
        # Track statistics
        self.move_count += 1
        self.thinking_time.append(time.time() - start_time)
        
        return row, col
    
    def _mask_invalid_moves(self, policy_logits: torch.Tensor, board: Board) -> torch.Tensor:
        """Mask invalid moves with -inf"""
        masked = policy_logits.clone()
        
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row][col] != '-':
                    idx = row * board.size + col
                    masked[0, idx] = -float('inf')
        
        return masked
    
    def _select_move(self, masked_policy: torch.Tensor) -> int:
        """Select move based on policy and temperature"""
        masked_policy = masked_policy.squeeze()
        
        if self.temperature == 0:
            # Deterministic: choose best move
            return masked_policy.argmax().item()
        else:
            # Stochastic: sample from distribution
            scaled = masked_policy / self.temperature
            probs = F.softmax(scaled, dim=0)
            
            # Handle numerical issues
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            
            return torch.multinomial(probs, 1).item()
    
    def get_move_probabilities(self, board: Board) -> np.ndarray:
        """Get raw move probabilities for analysis"""
        canonical = board.to_flat_canonical(self.token)
        state_tensor = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, policy_logits = self.model(state_tensor)
            masked = self._mask_invalid_moves(policy_logits, board)
            probs = F.softmax(masked, dim=1)
        
        return probs.squeeze().cpu().numpy()
    
    def get_value_estimate(self, board: Board) -> float:
        """Get value estimate for current position"""
        canonical = board.to_flat_canonical(self.token)
        state_tensor = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            value, _ = self.model(state_tensor)
        
        return value.item()
    
    def reset_stats(self):
        """Reset player statistics"""
        self.move_count = 0
        self.thinking_time = []
    
    def get_stats(self) -> dict:
        """Get player statistics"""
        stats = {
            'move_count': self.move_count,
            'avg_thinking_time': np.mean(self.thinking_time) if self.thinking_time else 0,
            'total_thinking_time': sum(self.thinking_time)
        }
        return stats


class MCTSNeuralPlayer(NeuralPlayer):
    """
    Neural player enhanced with Monte Carlo Tree Search.
    Combines neural network evaluation with tree search for stronger play.
    """
    
    def __init__(self, token: str, model, config=None, 
                 num_simulations: int = 100, c_puct: float = 1.0,
                 name: Optional[str] = None, device: str = 'cpu'):
        super().__init__(token, model, config, temperature=0.0, name=name, device=device)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def get_move(self, board: Board) -> Tuple[int, int]:
        """Get move using MCTS guided by neural network"""
        # Run MCTS simulations
        root = MCTSNode(board, self.token)
        
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Select best move based on visit counts
        best_child = max(root.children.values(), key=lambda n: n.visits)
        return best_child.move
    
    def _simulate(self, node: 'MCTSNode'):
        """Run one MCTS simulation"""
        path = [node]
        
        # Selection: traverse tree to leaf
        while node.is_expanded() and not node.is_terminal():
            node = self._select_child(node)
            path.append(node)
        
        # Expansion and Evaluation
        if not node.is_terminal():
            # Expand node
            value = self._expand_and_evaluate(node)
        else:
            # Terminal node evaluation
            result = node.board.get_game_result()
            if result == node.player_token:
                value = 1.0
            elif result == 'D':
                value = 0.0
            else:
                value = -1.0
        
        # Backup: update value estimates along path
        self._backup(path, value)
    
    def _select_child(self, node: 'MCTSNode') -> 'MCTSNode':
        """Select child using PUCT formula"""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                # Negate child value because it's from opponent's perspective
                exploitation = -child.value
                exploration = self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
    
    def _expand_and_evaluate(self, node: 'MCTSNode') -> float:
        """Expand node and get neural network evaluation"""
        # Get neural network evaluation
        canonical = node.board.to_flat_canonical(node.player_token)
        state_tensor = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            value, policy_logits = self.model(state_tensor)
            
        # Mask and normalize policy
        masked_policy = self._mask_invalid_moves(policy_logits, node.board)
        policy_probs = F.softmax(masked_policy, dim=1).squeeze().cpu().numpy()
        
        # Create child nodes
        valid_moves = node.board.get_valid_moves()
        for move in valid_moves:
            child_board = node.board.copy()
            child_board.make_move(move[0], move[1], node.player_token)
            
            # Switch player for child
            child_token = 'O' if node.player_token == 'X' else 'X'
            move_idx = move[0] * node.board.size + move[1]
            prior = policy_probs[move_idx]
            
            child = MCTSNode(child_board, child_token, move=move, prior=prior)
            node.children[move] = child
        
        node.expanded = True
        return value.item()
    
    def _backup(self, path: List['MCTSNode'], value: float):
        """Backup value estimates along path"""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            node.value = node.value_sum / node.visits
            value = -value  # Flip value for opponent


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board: Board, player_token: str, 
                 move: Optional[Tuple[int, int]] = None, prior: float = 0.0):
        self.board = board
        self.player_token = player_token
        self.move = move
        self.prior = prior
        
        self.visits = 0
        self.value = 0.0
        self.value_sum = 0.0
        self.children = {}
        self.expanded = False
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return self.expanded
    
    def is_terminal(self) -> bool:
        """Check if node is terminal (game over)"""
        return self.board.get_game_result() is not None


def create_player(player_type: str, token: str, model=None, config=None, **kwargs) -> Player:
    """
    Factory function to create different player types.
    
    Args:
        player_type: Type of player ('human', 'random', 'neural', 'mcts')
        token: Player token ('X' or 'O')
        model: Neural network model (for neural players)
        config: Configuration object
        **kwargs: Additional player-specific parameters
    """
    from game import HumanPlayer, RandomPlayer
    
    if player_type == 'human':
        return HumanPlayer(token, **kwargs)
    elif player_type == 'random':
        return RandomPlayer(token, **kwargs)
    elif player_type == 'neural':
        if model is None:
            raise ValueError("Neural player requires a model")
        return NeuralPlayer(token, model, config, **kwargs)
    elif player_type == 'mcts':
        if model is None:
            raise ValueError("MCTS player requires a model")
        return MCTSNeuralPlayer(token, model, config, **kwargs)
    else:
        raise ValueError(f"Unknown player type: {player_type}")
