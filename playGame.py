import chess
import chess.engine
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import traceback
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
from collections import defaultdict
import random
import chess.pgn
import datetime
import os
import re
from engines import RawNNEngine, MCTSEngine
from encoding import ChessEncoder
from neuralNetwork import ChessNN 
from mcts import MCTS

# ================================================================
# == Stockfish Configuration ==
# ================================================================
from dotenv import load_dotenv
load_dotenv() # Load .env file
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH") # Path to stockfish binary
try:
    STOCKFISH_THINK_TIME = float(os.getenv("STOCKFISH_THINK_TIME", "0.1")) 
    STOCKFISH_THREADS = int(os.getenv("STOCKFISH_THREADS", "12"))
except ValueError:
    print("Warning: Invalid STOCKFISH_THINK_TIME or STOCKFISH_THREADS in .env. Using defaults.")
    STOCKFISH_THINK_TIME = 0.1
    STOCKFISH_THREADS = 12

# ==============================================================================
# == ChessGameUI Class ==
# ==============================================================================
class ChessGameUI:
    """ Chess game UI using MCTSEngine """
    def __init__(self, model_state_dict, num_simulations: int = 400, exploration_weight: float = 1.4, stockfish_engine: Optional[chess.engine.SimpleEngine] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = ChessNN(input_channels=18)
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully for ChessGameUI.")
        except NameError: raise RuntimeError("ChessNN class not available.")
        except FileNotFoundError: raise FileNotFoundError("Model state dict file not found.")
        except Exception as e: raise RuntimeError(f"Error initializing model in UI: {e}")

        self.board = chess.Board()
        try:
            encoder_instance = ChessEncoder()
            # Pass the instantiated model, encoder, MCTS params, and stockfish engine
            self.engine = MCTSEngine(self.model,
                                      encoder=encoder_instance,
                                      num_simulations=num_simulations,
                                      exploration_weight=exploration_weight,
                                      stockfish_engine=stockfish_engine)
            print(f"MCTSEngine (MCTS) initialized for UI.")
        except RuntimeError as e:
            raise RuntimeError(f"Could not initialize MCTSEngine: {e}")
        self.move_history = []

    def make_engine_move(self):
        """ Make the best move using the MCTS engine and return it. """
        if self.board.is_game_over(): return None
        best_move, move_visits = self.engine.select_move(self.board)
        if best_move:
            print(f"Engine confirming play: {best_move.uci()}") 
            self.board.push(best_move)
            self.move_history.append(best_move)
            return best_move
        print("Engine (MCTS) could not find a move to confirm.")
        return None

    def make_player_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves: self.board.push(move); self.move_history.append(move); return True
        return False
    def get_legal_moves(self) -> List[chess.Move]: return list(self.board.legal_moves)
    def is_game_over(self): return self.board.is_game_over()
    def get_result(self): return self.board.result()
    def get_board(self): return self.board
    def undo_move(self) -> bool:
        if self.move_history: self.board.pop(); self.move_history.pop(); return True
        return False
    def reset_game(self): self.board.reset(); self.move_history = []


# ==============================================================================
# == Human vs MCTSEngine ==
# ==============================================================================

def play_human_vs_engine(model_state_dict, num_simulations: int, exploration_weight: float = 1.4, use_stockfish: bool = False):
     """ Manages human vs engine game loop using MCTS. """
     stockfish = None
     try:
        if use_stockfish:
            if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
                 try:
                    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
                    stockfish.configure({"Threads": STOCKFISH_THREADS})
                    print(f"Stockfish initialized at {STOCKFISH_PATH} for comparison.")
                 except Exception as sf_e:
                     print(f"Failed to initialize Stockfish: {sf_e}. Disabling comparison.")
                     stockfish = None
            else:
                 print("Stockfish path not found/configured. Disabling comparison.")

        game = ChessGameUI(model_state_dict,
                           num_simulations=num_simulations,
                           exploration_weight=exploration_weight,
                           stockfish_engine=stockfish) # Pass engine instance here
        model_name = "Engine (MCTS)"
        print(f"\nStarting game against {model_name}. You are White.")
        print(f"Engine using device: {game.engine.device}")

        while not game.is_game_over():
            print("\n" + "="*30); print(f"Move {game.board.fullmove_number}"); print(game.board)

            if game.board.turn == chess.WHITE: # Human's turn
                move_uci = input("Your move (UCI, e.g., e2e4), 'undo', or 'reset': ").strip()
                if move_uci.lower() == 'undo':
                    if game.undo_move():
                        if game.undo_move(): print("Two moves undone.")
                        else: print("Only engine move undone.")
                    else: print("Cannot undo further!")
                    continue
                elif move_uci.lower() == 'reset': game.reset_game(); print("Game reset."); continue
                try: move = chess.Move.from_uci(move_uci)
                except ValueError: print("Invalid move format! Use UCI."); continue
                if not game.make_player_move(move): print("Illegal move! Try again."); continue
            else: # Engine's turn (Black)
                print("Engine thinking (MCTS)...")
                engine_move = game.make_engine_move() # Calls select_move which prints comparison
                if not engine_move: print("Engine failed to make a move."); break

        print("\n--- Game Over ---"); print(game.board); print(f"Result: {game.get_result()}")

     except RuntimeError as rte: print(f"\nFATAL ERROR during game setup: {rte}"); traceback.print_exc()
     except Exception as e: print(f"\nAn error occurred during the game: {e}"); traceback.print_exc()
     finally:
        if stockfish: stockfish.quit(); print("Stockfish engine closed.")
     input("\nPress Enter to return to the menu...")

# ==============================================================================
# == MCTSEngine vs MCTSEngine ==
# ==============================================================================

def play_engine_vs_engine(state_dict1, state_dict2, num_simulations: int, exploration_weight: float = 1.4, use_stockfish=False, pause_after_move=False, model1_name="NN1", model2_name="NN2"):
     """ Manages game loop for two MCTS engines, saves the game as PGN. """
     stockfish = None
        # Create a directory to save the PGN files
     save_dir = "saved_engine_games"; os.makedirs(save_dir, exist_ok=True)
     try:
        # Check if the models are compatible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"\nEngines using device: {device}")

        if use_stockfish:
            if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
                 try: stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH); stockfish.configure({"Threads": STOCKFISH_THREADS}); print(f"Stockfish initialized at {STOCKFISH_PATH} for comparison.")
                 except Exception as sf_e: print(f"Failed to initialize Stockfish: {sf_e}. Disabling comparison."); stockfish = None
            else: print("Stockfish path not found/configured. Disabling comparison.")

        print(f"Instantiating {model1_name}..."); model1_instance = ChessNN(input_channels=18); model1_instance.load_state_dict(state_dict1); model1_instance.to(device); model1_instance.eval()
        print(f"Instantiating {model2_name}..."); model2_instance = ChessNN(input_channels=18); model2_instance.load_state_dict(state_dict2); model2_instance.to(device); model2_instance.eval()

        encoder_instance = ChessEncoder() # Create one encoder instance
        engine1 = MCTSEngine(model1_instance, encoder=encoder_instance, num_simulations=num_simulations, exploration_weight=exploration_weight, stockfish_engine=stockfish)
        engine2 = MCTSEngine(model2_instance, encoder=encoder_instance, num_simulations=num_simulations, exploration_weight=exploration_weight, stockfish_engine=stockfish)


        # Set up the game board
        board = chess.Board(); 
        print("\n--- Starting Engine vs Engine Game ---"); 
        print(f"Engine 1 (White): {model1_name}"); 
        print(f"Engine 2 (Black): {model2_name}"); 
        print(f"MCTS Simulations: {num_simulations} per move (c_puct={exploration_weight})\n")

        # Initialize PGN game
        game_pgn = chess.pgn.Game(); 
        game_pgn.headers["Event"] = "Engine Match (MCTS)"; 
        game_pgn.headers["Site"] = "Local Machine"; 
        game_pgn.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d"); 
        game_pgn.headers["Round"] = "1"; 
        game_pgn.headers["White"] = model1_name; 
        game_pgn.headers["Black"] = model2_name; 
        game_pgn.headers["MCTSSimulations"] = str(num_simulations); 
        current_pgn_node = game_pgn

        # Game loop
        while not board.is_game_over():
            # Print the board and move number
            print("\n" + "="*30); move_num_str = f"Move {board.fullmove_number}"; print(move_num_str if board.turn == chess.WHITE else move_num_str + "..."); print(board)
            # Select the engine based on the turn
            current_engine, engine_name = (engine1, f"Engine 1 (W - {model1_name})") if board.turn == chess.WHITE else (engine2, f"Engine 2 (B - {model2_name})")
            # Call the engine to select a move
            print(f"{engine_name} thinking (MCTS)...")
            best_move, move_visits = current_engine.select_move(board) # Calls MCTS + comparison inside
            if best_move:
                print(f"{engine_name} confirming play: {best_move.uci()}") # Simplified confirm
                board.push(best_move)
                current_pgn_node = current_pgn_node.add_variation(best_move) # Add to PGN
                if pause_after_move: input("Enter to continue...")
            else: print(f"{engine_name} (MCTS) failed to find a legal move!"); break

        print("\n--- Game Over ---"); print(board); game_result = board.result(); print(f"Result: {game_result}"); game_pgn.headers["Result"] = game_result

        # Save the game as PGN
        safe_m1 = re.sub(r'[\\/*?:"<>|]', "", model1_name.replace('.pth','')); 
        safe_m2 = re.sub(r'[\\/*?:"<>|]', "", model2_name.replace('.pth','')); 
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); 
        pgn_filename = f"Match_{safe_m1}_vs_{safe_m2}_{timestamp}.pgn"; 
        pgn_filepath = os.path.join(save_dir, pgn_filename)
        print(f"\nSaving game to {pgn_filepath}...")
        try:
            with open(pgn_filepath, "w", encoding="utf-8") as f: exporter = chess.pgn.FileExporter(f); game_pgn.accept(exporter)
            print("Game saved successfully.")
        except Exception as e: print(f"Error saving PGN file: {e}")

     except RuntimeError as rte: print(f"\nFATAL ERROR during engine setup: {rte}"); traceback.print_exc()
     except KeyboardInterrupt: print("\nGame interrupted by user.")
     except Exception as e: print(f"\nAn error occurred during the engine vs engine game: {e}"); traceback.print_exc()
     finally:
         if stockfish: stockfish.quit(); print("Stockfish engine closed.")
     input("\nPress Enter to return to the menu...")


# ==============================================================================
# == Raw NN vs MCTS engines ==
# ==============================================================================
def play_raw_vs_mcts(state_dict, num_simulations: int, exploration_weight: float = 1.4, raw_plays_white: bool = True, use_stockfish=False, pause_after_move=False, model_name="NN"):
     """ Manages game loop for Raw NN vs MCTS engine using the same model. """
     stockfish = None
    # Create a directory to save the PGN files
     save_dir = "saved_comparison_games"; os.makedirs(save_dir, exist_ok=True)

     try:
        # Check if the models are compatible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"\nEngine using device: {device}")
        if use_stockfish:
            if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
                 try: stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH); stockfish.configure({"Threads": STOCKFISH_THREADS}); print(f"Stockfish initialized at {STOCKFISH_PATH} for comparison.")
                 except Exception as sf_e: print(f"Failed to initialize Stockfish: {sf_e}. Disabling comparison."); stockfish = None
            else: print("Stockfish path not found/configured. Disabling comparison.")

        print(f"Instantiating model {model_name}..."); model_instance = ChessNN(input_channels=18); model_instance.load_state_dict(state_dict); model_instance.to(device); model_instance.eval()
        encoder_instance = ChessEncoder() # Create one encoder

        print("Creating Raw NN Engine..."); raw_engine = RawNNEngine(model_instance, encoder=encoder_instance)
        print("Creating MCTS Engine..."); mcts_engine = MCTSEngine(model_instance, encoder=encoder_instance, num_simulations=num_simulations, exploration_weight=exploration_weight, stockfish_engine=stockfish)

        # Set up the engines based on the color         
        if raw_plays_white: engine1, engine2 = raw_engine, mcts_engine; name1, name2 = f"RawNN ({model_name})", f"MCTS ({model_name})"
        else: engine1, engine2 = mcts_engine, raw_engine; name1, name2 = f"MCTS ({model_name})", f"RawNN ({model_name})"

        # Set up the game board
        board = chess.Board(); 
        print("\n--- Starting Raw NN vs MCTS Game ---"); 
        print(f"Engine 1 (White): {name1}"); 
        print(f"Engine 2 (Black): {name2}"); 
        print(f"MCTS Simulations (for MCTS engine): {num_simulations} per move (c_puct={exploration_weight})\n")

        # Initialize PGN game
        game_pgn = chess.pgn.Game(); 
        game_pgn.headers["Event"] = "Raw NN vs MCTS Comparison"; 
        game_pgn.headers["Site"] = "Local Machine"; 
        game_pgn.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d"); 
        game_pgn.headers["Round"] = "1"; game_pgn.headers["White"] = name1; 
        game_pgn.headers["Black"] = name2; game_pgn.headers["MCTSSimulations"] = str(num_simulations); 
        current_pgn_node = game_pgn

        # Game loop
        while not board.is_game_over():
            print("\n" + "="*30); 
            move_num_str = f"Move {board.fullmove_number}"; 
            print(move_num_str if board.turn == chess.WHITE else move_num_str + "..."); 
            print(board)
            current_engine, engine_name = (engine1, name1) if board.turn == chess.WHITE else (engine2, name2)
            print(f"{engine_name} thinking...")
            best_move, move_stats = current_engine.select_move(board) # Calls appropriate select_move

            if best_move:
                # Print confirmation based on engine type
                if isinstance(current_engine, RawNNEngine):
                     prob = move_stats.get(best_move, 0.0)
                     print(f"{engine_name} confirming play: {best_move.uci()} (Raw NN Prob: {prob:.4f})")
                else: # MCTS engine
                     visits = move_stats.get(best_move, 0)
                     # Q value isn't returned directly here, but printed inside select_move
                     print(f"{engine_name} confirming play: {best_move.uci()} (MCTS Visits: {visits})")

                board.push(best_move)
                current_pgn_node = current_pgn_node.add_variation(best_move)

                # Optional Stockfish analysis AFTER the move for the RawNN player
                # (MCTS engine already does this comparison inside its select_move)
                if isinstance(current_engine, RawNNEngine) and stockfish:
                     try:
                         limit = chess.engine.Limit(time=STOCKFISH_THINK_TIME)
                         info = stockfish.analyse(board, limit) # Analyse board AFTER raw NN move
                         # Use the helper method from the RawNNEngine instance
                         sf_eval_str = current_engine._get_stockfish_score_str(info)
                         if "N/A" not in sf_eval_str and "Err" not in sf_eval_str:
                             print(f"  Stockfish eval after {best_move.uci()}: {sf_eval_str.replace('SF: ','')}")
                     except Exception as sf_eval_e: print(f"SF Error evaluating after raw NN move: {sf_eval_e}")

                if pause_after_move: input("Enter to continue...")
            else: print(f"{engine_name} failed to find a legal move!"); break

        print("\n--- Game Over ---"); print(board); game_result = board.result(); print(f"Result: {game_result}"); game_pgn.headers["Result"] = game_result

        # Save the game as PGN
        safe_m = re.sub(r'[\\/*?:"<>|]', "", model_name.replace('.pth','')); 
        w_player = "RawNN" if raw_plays_white else "MCTS"; 
        b_player = "MCTS" if raw_plays_white else "RawNN"; 
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); 
        pgn_filename = f"Compare_{w_player}_vs_{b_player}_{safe_m}_{timestamp}.pgn"; 
        pgn_filepath = os.path.join(save_dir, pgn_filename)
        print(f"\nSaving comparison game to {pgn_filepath}...")
        try:
            with open(pgn_filepath, "w", encoding="utf-8") as f: exporter = chess.pgn.FileExporter(f); game_pgn.accept(exporter)
            print("Game saved successfully.")
        except Exception as e: print(f"Error saving PGN file: {e}")

     except RuntimeError as rte: print(f"\nFATAL ERROR during game setup: {rte}"); traceback.print_exc()
     except KeyboardInterrupt: print("\nGame interrupted by user.")
     except Exception as e: print(f"\nAn error occurred during the Raw NN vs MCTS game: {e}"); traceback.print_exc()
     finally:
        if stockfish: stockfish.quit(); print("Stockfish engine closed.")
     input("\nPress Enter to return to the menu...")