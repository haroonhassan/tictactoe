"""
Main script for training and evaluating Tic-Tac-Toe neural network.
Provides command-line interface and orchestrates the training process.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path
import json
import sys

from config import Config, get_quick_test_config, get_production_config, get_debug_config
from models import create_model, load_checkpoint
from trainer import Trainer
from logger import Logger
from game import Board, Game, HumanPlayer
from neural_players import NeuralPlayer, MCTSNeuralPlayer, create_player


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """Train a new model or continue training an existing one"""
    # Load or create configuration
    if args.config_file:
        config = Config.load(args.config_file)
        print(f"Loaded configuration from {args.config_file}")
    elif args.config_preset:
        if args.config_preset == 'quick':
            config = get_quick_test_config()
        elif args.config_preset == 'production':
            config = get_production_config()
        elif args.config_preset == 'debug':
            config = get_debug_config()
        else:
            raise ValueError(f"Unknown preset: {args.config_preset}")
        print(f"Using {args.config_preset} configuration preset")
    else:
        config = Config()
        print("Using default configuration")
    
    # Override config with command-line arguments
    if args.num_games:
        config.training.num_games = args.num_games
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.hidden_size:
        config.model.hidden_size = args.hidden_size
    
    # Set seed
    if config.seed:
        set_seed(config.seed)
    
    # Create or load model
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, config)
        print(f"Resumed from checkpoint: {args.resume}")
        start_game = checkpoint.get('game_num', 0)
    else:
        model = create_model(config, args.model_type)
        print(f"Created new {args.model_type} model")
        start_game = 0
    
    # Create logger
    logger = Logger(config)
    
    # Create trainer
    trainer = Trainer(model, config, logger)
    
    # Resume if needed
    if args.resume:
        trainer.game_count = start_game
    
    # Save configuration
    config_path = Path(logger.log_dir) / "config.json"
    config.save(str(config_path))
    print(f"Configuration saved to {config_path}")
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Training games: {config.training.num_games}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Logs saved to: {logger.log_dir}")


def evaluate(args):
    """Evaluate a trained model"""
    # Load model and config
    model, checkpoint = load_checkpoint(args.model_path)
    config = checkpoint.get('config', Config())
    
    print(f"Loaded model from {args.model_path}")
    print(f"Model trained for {checkpoint.get('game_num', 0)} games")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    
    # Create players
    if args.player1_type == 'neural':
        player1 = NeuralPlayer('X', model, config, temperature=args.temperature, device=device)
    elif args.player1_type == 'mcts':
        player1 = MCTSNeuralPlayer('X', model, config, num_simulations=args.mcts_simulations, device=device)
    else:
        player1 = create_player(args.player1_type, 'X')
    
    if args.player2_type == 'neural':
        player2 = NeuralPlayer('O', model, config, temperature=args.temperature, device=device)
    elif args.player2_type == 'mcts':
        player2 = MCTSNeuralPlayer('O', model, config, num_simulations=args.mcts_simulations, device=device)
    else:
        player2 = create_player(args.player2_type, 'O')
    
    # Run evaluation games
    results = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
    game = Game(verbose=args.verbose)
    
    print(f"\nPlaying {args.num_games} games: {player1.name} vs {player2.name}")
    print("-" * 40)
    
    for i in range(args.num_games):
        # Alternate who goes first
        if i % 2 == 0:
            result = game.play(player1, player2)
            if result['winner'] == 'X':
                results['player1_wins'] += 1
            elif result['winner'] == 'O':
                results['player2_wins'] += 1
            else:
                results['draws'] += 1
        else:
            result = game.play(player2, player1)
            if result['winner'] == 'O':
                results['player1_wins'] += 1
            elif result['winner'] == 'X':
                results['player2_wins'] += 1
            else:
                results['draws'] += 1
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{args.num_games} games")
    
    # Print results
    total = args.num_games
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"{player1.name} wins: {results['player1_wins']} ({results['player1_wins']/total:.1%})")
    print(f"{player2.name} wins: {results['player2_wins']} ({results['player2_wins']/total:.1%})")
    print(f"Draws: {results['draws']} ({results['draws']/total:.1%})")
    print("="*40)


def play(args):
    """Play against a trained model"""
    # Load model
    model, checkpoint = load_checkpoint(args.model_path)
    config = checkpoint.get('config', Config())
    
    print(f"Loaded model from {args.model_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    
    # Create players
    if args.ai_type == 'neural':
        ai_player = NeuralPlayer('O', model, config, temperature=args.temperature, 
                                name="AI", device=device)
    else:  # mcts
        ai_player = MCTSNeuralPlayer('O', model, config, 
                                     num_simulations=args.mcts_simulations,
                                     name="AI (MCTS)", device=device)
    
    human_player = HumanPlayer('X', name="Human")
    
    # Game loop
    game = Game(verbose=True)
    
    print("\n" + "="*50)
    print("TIC-TAC-TOE: Human vs AI")
    print("="*50)
    print("You are X, AI is O")
    print("Enter moves as: row,col (e.g., 2,3)")
    print("="*50)
    
    while True:
        # Determine who goes first
        if args.human_first:
            result = game.play(human_player, ai_player)
        else:
            result = game.play(ai_player, human_player)
        
        # Ask if want to play again
        play_again = input("\nPlay again? (y/n): ")
        if play_again.lower() != 'y':
            break
        
        # Switch who goes first
        args.human_first = not args.human_first
        print(f"\n{'Human' if args.human_first else 'AI'} goes first this time!")
    
    print("\nThanks for playing!")


def analyze(args):
    """Analyze a model's behavior"""
    # Load model
    model, checkpoint = load_checkpoint(args.model_path)
    config = checkpoint.get('config', Config())
    
    print(f"Loaded model from {args.model_path}")
    print(model.get_architecture_summary())
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    
    # Create neural player for analysis
    player = NeuralPlayer('X', model, config, temperature=0, device=device)
    
    # Analyze specific positions
    print("\n" + "="*50)
    print("POSITION ANALYSIS")
    print("="*50)
    
    # Empty board
    board = Board()
    value = player.get_value_estimate(board)
    probs = player.get_move_probabilities(board)
    
    print("\nEmpty board:")
    board.display()
    print(f"Value estimate: {value:.3f}")
    print("Move probabilities:")
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            print(f"  ({i+1},{j+1}): {probs[idx]:.3f}", end="  ")
        print()
    
    # Center occupied
    board = Board()
    board.make_move(1, 1, 'X')
    canonical_o = board.to_canonical('O')
    board_for_o = Board()
    board_for_o.grid = [['-' if cell == 0 else ('O' if cell == 1 else 'X') 
                        for cell in row] for row in canonical_o]
    
    value = player.get_value_estimate(board_for_o)
    probs = player.get_move_probabilities(board_for_o)
    
    print("\nAfter X plays center (from O's perspective):")
    board.display()
    print(f"Value estimate for O: {value:.3f}")
    print("Move probabilities for O:")
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if board.grid[i][j] == '-':
                print(f"  ({i+1},{j+1}): {probs[idx]:.3f}", end="  ")
        print()
    
    # Test perfect play scenario
    if args.test_perfect:
        print("\n" + "="*50)
        print("TESTING PERFECT PLAY")
        print("="*50)
        
        # Create MCTS player for stronger play
        mcts_player = MCTSNeuralPlayer('X', model, config, 
                                       num_simulations=100, device=device)
        opponent = MCTSNeuralPlayer('O', model, config, 
                                    num_simulations=100, device=device)
        
        game = Game()
        wins = draws = losses = 0
        
        for i in range(10):
            if i % 2 == 0:
                result = game.play(mcts_player, opponent)
                if result['winner'] == 'X':
                    wins += 1
                elif result['winner'] == 'O':
                    losses += 1
                else:
                    draws += 1
            else:
                result = game.play(opponent, mcts_player)
                if result['winner'] == 'O':
                    wins += 1
                elif result['winner'] == 'X':
                    losses += 1
                else:
                    draws += 1
        
        print(f"Self-play with MCTS (10 games):")
        print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"  Perfect play should result in all draws!")


def main():
    parser = argparse.ArgumentParser(description='Tic-Tac-Toe Neural Network Training')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config-file', type=str, help='Path to config file')
    train_parser.add_argument('--config-preset', choices=['quick', 'production', 'debug'],
                            help='Use a preset configuration')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--model-type', default='standard', 
                            choices=['standard', 'resnet'],
                            help='Model architecture type')
    train_parser.add_argument('--num-games', type=int, help='Override number of games')
    train_parser.add_argument('--batch-size', type=int, help='Override batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    train_parser.add_argument('--hidden-size', type=int, help='Override hidden size')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('model_path', help='Path to model checkpoint')
    eval_parser.add_argument('--num-games', type=int, default=100, 
                           help='Number of evaluation games')
    eval_parser.add_argument('--player1-type', default='neural',
                           choices=['neural', 'mcts', 'random', 'human'],
                           help='Type of player 1')
    eval_parser.add_argument('--player2-type', default='random',
                           choices=['neural', 'mcts', 'random', 'human'],
                           help='Type of player 2')
    eval_parser.add_argument('--temperature', type=float, default=0.0,
                           help='Temperature for neural players')
    eval_parser.add_argument('--mcts-simulations', type=int, default=100,
                           help='Number of MCTS simulations')
    eval_parser.add_argument('--verbose', action='store_true', 
                           help='Show game progress')
    eval_parser.add_argument('--cpu', action='store_true',
                           help='Force CPU usage')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the model')
    play_parser.add_argument('model_path', help='Path to model checkpoint')
    play_parser.add_argument('--ai-type', default='neural', 
                           choices=['neural', 'mcts'],
                           help='AI player type')
    play_parser.add_argument('--temperature', type=float, default=0.0,
                           help='Temperature for AI moves')
    play_parser.add_argument('--mcts-simulations', type=int, default=100,
                           help='Number of MCTS simulations')
    play_parser.add_argument('--human-first', action='store_true',
                           help='Human plays first')
    play_parser.add_argument('--cpu', action='store_true',
                           help='Force CPU usage')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model behavior')
    analyze_parser.add_argument('model_path', help='Path to model checkpoint')
    analyze_parser.add_argument('--test-perfect', action='store_true',
                              help='Test for perfect play')
    analyze_parser.add_argument('--cpu', action='store_true',
                              help='Force CPU usage')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'play':
        play(args)
    elif args.command == 'analyze':
        analyze(args)


if __name__ == '__main__':
    main()
