import chess
import torch
import time
import torch.nn.functional as F
from typing import List

# Import ChessEncoder and ChessNN
from encoding import ChessEncoder
from neuralNetwork import ChessNN

class ChessEngine:
    """Chess engine that uses the given neural network model for move selection."""
    
    def __init__(self, model, think_time=2.0):
        """
        Initialize the chess engine with a model.
        
        Args:
            model: The PyTorch model instance
            think_time: Time in seconds given to the engine to think about a given move
        """
        self.model = model
        self.think_time = think_time
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU used for move inference
        self.encoder = ChessEncoder()
        
        # Debug device check
        #print(f"Using device: {self.device} with mixed precision inference")
    
    def select_move(self, board: chess.Board):
        """
        (Try to) Find the best move for a given position
        Soon to be replaced by MCTS approach from inference.py
        
        Args:
            board: The current chess board
            
        Returns:
            Tuple with the best move and a dictionary of probabilities
        """
        start_time = time.time()
        moves_evaluated = {}  # Track evaluations for each move
        
        with torch.no_grad():
            # Check positions until time runs out
            evaluations_done = 0
            while (time.time() - start_time) < self.think_time: # While time left
                # Encode board
                encoded_position = self.encoder.encode_board(board)
                position = torch.FloatTensor(encoded_position).unsqueeze(0).to(self.device)
                
                # Run inference with mixed precision
                with torch.amp.autocast(device_type=self.device.type):
                    policy, value = self.model(position)
                
                # Process move probabilities
                move_probs = F.softmax(policy, dim=1).squeeze().cpu()
                
                # Update evaluations for legal moves
                for move in board.legal_moves:
                    move_idx = self.encoder.encode_move(move)
                    
                    # Skip moves that are out of range for our model's output
                    if move_idx >= move_probs.shape[0]:
                        continue
                        
                    score = move_probs[move_idx].item()
                    
                    if move in moves_evaluated:
                        moves_evaluated[move] = (
                            moves_evaluated[move][0] + score,
                            moves_evaluated[move][1] + 1
                        )
                    else:
                        moves_evaluated[move] = (score, 1)
                
                evaluations_done += 1
        
        # Get average of positions evaluated (no tree search - same positions get evaluated multiple times)
        move_scores = {}
        best_move = None
        best_score = -float('inf')
        
        for move, (total_score, count) in moves_evaluated.items():
            avg_score = total_score / count
            move_scores[move] = avg_score
            
            if avg_score > best_score:
                best_score = avg_score
                best_move = move
        
        # If no legal moves were found, return a random legal move
        if best_move is None and len(list(board.legal_moves)) > 0:
            print("No legal moves found, selecting a random move")
            best_move = next(iter(board.legal_moves))
            move_scores[best_move] = 1.0
        
        # Debug log info - time taken + positions evaluated
        elapsed = time.time() - start_time
        print(f"Thought for {elapsed:.2f}s, evaluated {evaluations_done} positions")
        
        return best_move, move_scores

    def get_position_evaluation(self, board: chess.Board):
        """Get the model's evaluation of the current position."""
        with torch.no_grad():
            encoded_position = self.encoder.encode_board(board)
            position = torch.FloatTensor(encoded_position).unsqueeze(0).to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type):
                _, value = self.model(position)
            
            # Return value as a float between -1 and 1
            return value.item()


class ChessGameUI:
    """
    Chess game UI integration that uses the ChessEngine for move selection.
    """
    
    def __init__(self, model_state_dict, think_time=2.0):
        """
        Initialize the chess game UI with a pre-loaded model state dictionary.
        
        Args:
            model_state_dict: Pre-loaded model state dictionary
            think_time: Time in seconds to think about each move
        """
        # Create an instance of your model and load the state dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNN(input_channels=18)  # 18 channels for the encoder - input plane count
        self.model.load_state_dict(model_state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Initialize the board and engine
        self.board = chess.Board()
        self.engine = ChessEngine(self.model, think_time=think_time)
        self.move_history = []
    
    def make_engine_move(self):
        """Make the best move with a given engine and return it"""
        if self.board.is_game_over():
            return None
            
        best_move, move_scores = self.engine.select_move(self.board)
        
        if best_move:
            print(f"Engine plays: {best_move}, confidence: {move_scores[best_move]:.4f}")
            
            self.board.push(best_move)
            self.move_history.append(best_move)
            return best_move
        
        return None
    
    def make_player_move(self, move: chess.Move) -> bool:
        """
        Make a player move.
        
        Args:
            move: Chess move to make
            
        Returns:
            True if move was successful, False otherwise
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for the current position."""
        return list(self.board.legal_moves)
    
    def is_game_over(self):
        return self.board.is_game_over()
    
    def get_result(self):
        return self.board.result()
    
    def get_board(self):
        return self.board
    
    def undo_move(self) -> bool:
        """Undo the last move"""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            return True
        return False
    
    def reset_game(self):
        self.board.reset()
        self.move_history = []