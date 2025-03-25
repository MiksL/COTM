import chess
import random
import numpy as np
import torch
from stockfish import Stockfish
from tqdm import tqdm
from dotenv import load_dotenv
import os
import statistics

# Benchmark class implementation to test model against stockfish engine moves on a randomly generated board, taking the delta between the engine and stockfish moves with evaluations deltas taken from stockfish evaluations
class Benchmark:
    def __init__(self, model_dict, encoder, depth=10):
        """
        Benchmark to compare neural network model against Stockfish.
        
        Args:
            model_dict: Neural network model dictionary to evaluate 
            encoder: Encoder class used to encode chess positions
            depth: Stockfish analysis depth
        """
        # Load environment variables
        load_dotenv()
        
        # Get Stockfish path from .env
        stockfish_path = os.getenv("STOCKFISH_PATH")
        if not stockfish_path:
            raise ValueError("STOCKFISH_PATH environment variable not set")
        
        # Initialize model
        device = torch.device("cpu")
        from neuralNetwork import ChessNN
        self.model = ChessNN(input_channels=18)  # 18 channels for the encoder - input plane count
        self.model.load_state_dict(model_dict)
        self.model.to(device)
        
        self.model.eval()
        self.encoder = encoder
        
        # Initialize Stockfish
        try:
            self.stockfish = Stockfish(path=stockfish_path, depth=depth)
            print(f"Stockfish engine initialized")
        except Exception as e:
            raise Exception(f"Failed to initialize Stockfish: {e}")
    
    def generate_random_position(self, min_moves=10, max_moves=40):
        """Generate a random chess position with a random number of moves done by both sides"""
        board = chess.Board()
        num_moves = random.randint(min_moves, max_moves)
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        return board
    
    def get_model_move(self, board):
        """Get the best move from the neural network"""
        encoded_board = self.encoder.encode_board(board)
        board_tensor = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.model(board_tensor)
        
        # Get probabilities and legal moves
        policy = torch.softmax(policy, dim=1).squeeze().numpy()
        legal_moves = list(board.legal_moves)
        legal_move_indices = [self.encoder.encode_move(move) for move in legal_moves]
        
        # Find move with highest probability
        best_idx = np.argmax([policy[idx] for idx in legal_move_indices])
        return legal_moves[best_idx]
    
    def get_stockfish_evaluation(self, fen, move):
        """Get Stockfish's evaluation after a move"""
        # Set position and make move
        self.stockfish.set_fen_position(fen)
        move_str = move.uci()
        if self.stockfish.is_move_correct(move_str):
            self.stockfish.make_moves_from_current_position([move_str])
            evaluation = self.stockfish.get_evaluation()
            
            # Debug - output the evaluation
            #print(f"Stockfish evaluation: {evaluation}")
            #input("debug")
            
            # Handle mate scores 
            if 'mate' in evaluation:
                mate_in = evaluation['mate']
                if mate_in > 0:  # Checkmate for side to move
                    return 10000 - mate_in * 100  # Higher score for faster mate
                else:
                    return -10000 - mate_in * 100  # Loweer score when mated
            
            # Return given evaluation as centipawns
            return evaluation['value']
        else:
            # Fall back on invalid move
            return 0
    
    def run_benchmark(self, num_positions=100, min_moves=10, max_moves=40):
        """Run the benchmark comparison"""
        abs_diffs = []
        rel_diffs = []
        model_evals = []
        stockfish_evals = []
        
        # Track positions where model chose same as Stockfish
        same_move_count = 0
        
        for _ in tqdm(range(num_positions), desc="Benchmarking"):
            try:
                # Generate a random position with a random number of moves
                board = self.generate_random_position(min_moves, max_moves)
                if board.is_game_over():
                    continue
                
                # Get move from model
                model_move = self.get_model_move(board)
                
                # Get move from Stockfish
                fen = board.fen()
                self.stockfish.set_fen_position(fen)
                stockfish_move_str = self.stockfish.get_best_move()
                if not stockfish_move_str:
                    continue
                
                stockfish_move = chess.Move.from_uci(stockfish_move_str)
                
                # Check if model chose same as Stockfish
                if model_move == stockfish_move:
                    same_move_count += 1
                
                # Evaluate both moves
                model_eval = self.get_stockfish_evaluation(fen, model_move)
                stockfish_eval = self.get_stockfish_evaluation(fen, stockfish_move)

                abs_diff = abs(model_eval - stockfish_eval)
                
                #print(f"Model: {model_eval}, Stockfish: {stockfish_eval}, Diff: {abs_diff}")
                #input("debug")
                
                # Calculate relative difference
                if abs(stockfish_eval) > 100:
                    rel_diff = abs_diff / abs(stockfish_eval) * 100
                else:
                    rel_diff = abs_diff
                
                # Store results
                abs_diffs.append(abs_diff)
                rel_diffs.append(rel_diff)
                model_evals.append(model_eval)
                stockfish_evals.append(stockfish_eval)
                
            except Exception as e:
                print(f"Error analyzing position: {e}")
                continue
        
        # Calculate statistics
        positions_evaluated = len(abs_diffs)
        if positions_evaluated == 0:
            return {"error": "No positions were successfully evaluated"}
        
        return { # cp - centipawns
            "positions_evaluated": positions_evaluated, # number of positions evaluated
            "avg_abs_diff_cp": sum(abs_diffs) / positions_evaluated,  # average absolute difference
            "median_abs_diff_cp": statistics.median(abs_diffs), # median absolute difference
            "avg_rel_diff_percent": sum(rel_diffs) / positions_evaluated, # average relative difference
            "same_move_rate": same_move_count / positions_evaluated * 100, # percentage of same moves
            "avg_model_eval_cp": sum(model_evals) / positions_evaluated, # average model evaluation in centipawns
            "avg_stockfish_eval_cp": sum(stockfish_evals) / positions_evaluated # average Stockfish evaluation in centipawns
        }