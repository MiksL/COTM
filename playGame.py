import chess
import chess.engine
import torch
import time
import torch.nn.functional as F
import traceback
from typing import List

# Import ChessEncoder and ChessNN
from encoding import ChessEncoder
from neuralNetwork import ChessNN

# Get Stockfish path
from dotenv import load_dotenv
import os
load_dotenv() # Load .env file
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH") # Path to stockfish binary
STOCKFISH_THINK_TIME = 0.2
STOCKFISH_THREADS = 12

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
        self.model.eval() # Set to eval mode
        
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
    (Mainly for human vs engine play)
    """

    def __init__(self, model_state_dict, think_time=2.0):
        """
        Initialize the chess game UI with a pre-loaded model state dictionary.

        Args:
            model_state_dict: Pre-loaded model state dictionary
            think_time: Time in seconds to think about each move
        """
        # Create an instance of the model and load the state dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNN(input_channels=18)  # 18 channels for the encoder - input plane count
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval() # Ensure eval mode

        # Initialize the board and engine
        self.board = chess.Board()
        # Pass the instantiated model to the engine
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

def play_human_vs_engine(model_state_dict, think_time):
    """
    Manages the game loop for a human player against an engine.

    Args:
        model_state_dict: Pre-loaded state dictionary for the engine's model.
        think_time: Time in seconds for the engine to think.
    """
    try:
        # ChessGameUI handles model instantiation, device placement, and engine creation
        game = ChessGameUI(model_state_dict, think_time=think_time)
        model_name = "Engine" # Could pass model name if needed for display

        print(f"\nStarting game against {model_name}. You are White.")
        print(f"Engine using device: {game.engine.device}") # Show device used by engine

        # --- Human vs Engine Game Loop ---
        while not game.is_game_over():
            print("\n" + "="*30)
            print(f"Move {game.board.fullmove_number}")
            print(game.board)
            # Optional: Display engine's evaluation
            # eval_score = game.engine.get_position_evaluation(game.board)
            # print(f"Position Evaluation: {eval_score:.3f}")

            if game.board.turn == chess.WHITE:
                # Handle player move via UI
                move_uci = input("Your move (UCI, e.g., e2e4), 'undo', or 'reset': ").strip()

                if move_uci.lower() == 'undo':
                    if game.undo_move(): # Undo engine move
                        if game.undo_move(): # Undo player move
                             print("Two moves undone.")
                        else:
                             print("Only engine move undone (was at start?).") # Should be rare
                    else:
                        print("Cannot undo further!")
                    continue
                elif move_uci.lower() == 'reset':
                    game.reset_game()
                    print("Game reset.")
                    continue

                try:
                    move = chess.Move.from_uci(move_uci)
                    if game.make_player_move(move):
                        pass # Move successful
                    else:
                        print("Illegal move! Try again.")
                        continue # Let player try again
                except ValueError:
                    print("Invalid move format! Use UCI (e.g., 'e2e4').")
                    continue
                except Exception as e:
                     print(f"An error occurred processing your move: {e}")
                     continue
            else: # Engine's turn (Black)
                print("Engine thinking...")
                engine_move = game.make_engine_move()
                if not engine_move:
                   print("Engine failed to make a move.")
                   break

        print("\n--- Game Over ---")
        print(game.board)
        print(f"Result: {game.get_result()}")
        
        # Output Centipawn value of each side after game
        white_eval = game.engine.get_position_evaluation(game.board)
        black_eval = -white_eval
        print(f"White Evaluation: {white_eval:.3f} (Centipawns)")
        print(f"Black Evaluation: {black_eval:.3f} (Centipawns)")

    except Exception as e:
        print(f"\nAn error occurred during the game: {e}")
        traceback.print_exc()

    input("\nPress Enter to return to the menu...")


def play_engine_vs_engine(state_dict1, state_dict2, think_time, use_stockfish = False, pause_after_move = False, model1_name = None, model2_name = None):
    """
    Manages the game for two engines playing against each other.

    Args:
        state_dict1: Pre-loaded state dictionary for model 1 (White)
        state_dict2: Pre-loaded state dictionary for model 2 (Black)
        think_time: Time in seconds for each engine to think on a per move basis
        use_stockfish: Use Stockfish to get move evaluations if True
        pause_after_move: Pause after each move
        model1_name: Display name of model 1.
        model2_name: Display name of model 2.
    """
    
    stockfish = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nEngines using device: {device}")

        # --- Instantiate Models ---
        print(f"Instantiating {model1_name}...")
        model1_instance = ChessNN(input_channels=18) # Ensure input_channels
        model1_instance.load_state_dict(state_dict1)
        model1_instance.to(device)
        model1_instance.eval() # ChessEngine constructor calls eval()

        print(f"Instantiating {model2_name}...")
        model2_instance = ChessNN(input_channels=18) # Ensure input_channels
        model2_instance.load_state_dict(state_dict2)
        model2_instance.to(device)
        model2_instance.eval() # ChessEngine constructor calls eval()

        # Engine setup
        engine1 = ChessEngine(model1_instance, think_time=think_time)
        engine2 = ChessEngine(model2_instance, think_time=think_time)
        
        # Stockfish setup if True
        if use_stockfish:
            if not os.path.exists(STOCKFISH_PATH):
                print("No Stockfish engine found, skipping Stockfish evaluation.")
            else:
                stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
                print(f"Stockfish initialized at {STOCKFISH_PATH}.")
                stockfish.configure({"Threads": STOCKFISH_THREADS}) # Thread setup for Stockfish

        board = chess.Board()
        print("\n--- Starting Game ---")
        print(f"Engine 1 (White): {model1_name}")
        print(f"Engine 2 (Black): {model2_name}")
        print(f"Think Time: {think_time}s per move\n")

        # --- Engine vs Engine Game Loop ---
        while not board.is_game_over():
            print("\n" + "="*30)
            move_num_str = f"Move {board.fullmove_number}"
            print(move_num_str if board.turn == chess.WHITE else move_num_str + "...")
            print(board)

            current_engine = None
            engine_name = ""

            if board.turn == chess.WHITE:
                current_engine = engine1
                engine_name = f"Engine 1 (White - {model1_name})"
            else: # Black's turn
                current_engine = engine2
                engine_name = f"Engine 2 (Black - {model2_name})"

            print(f"{engine_name} thinking...")
            best_move, move_scores = current_engine.select_move(board)

            if best_move:
                move_confidence = move_scores.get(best_move, -1.0)
                print(f"{engine_name} plays: {best_move.uci()} \n Confidence: {move_confidence:.4f}")
                board.push(best_move)
                 
                if stockfish:
                    # Get time limit
                    print("Stockfish evaluation:")
                    time_limit = chess.engine.Limit(time=STOCKFISH_THINK_TIME)
                    
                    info = stockfish.analyse(board, time_limit)
                    score = info.get("score")
                    
                    # Score relative to white
                    white_score = score.white()
                    
                    if white_score.is_mate():
                        mate_in = white_score.mate()
                        print(f"Mate in {abs(mate_in)} for {'White' if mate_in > 0 else 'Black'}")
                    else:
                        cp = white_score.score()
                        # Convert to centipawns
                        cp = cp / 100.0 if cp else 0
                        print(f"Centipawn score: {cp}")
                
                if pause_after_move:
                    input("Enter to continue to next move...")   
            else:
                print(f"{engine_name} failed to find a legal move!")
                break # Exit loop

            #time.sleep(0.2) # Delay between moves

        # --- Game End ---
        print("\n--- Game Over ---")
        print(board)
        print(f"Result: {board.result()}")

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during the engine vs engine game: {e}")
        traceback.print_exc()

    input("\nPress Enter to return to the menu...")