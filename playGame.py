import chess
import chess.engine
import torch
import time
import traceback
from typing import List, Optional, Dict, Tuple
import numpy as np # Retained for MCTS if it uses it internally
import math # Retained for MCTS or other calcs, and for engine_utils helpers
from collections import defaultdict # Retained for MCTS
import random # Retained for MCTS
import chess.pgn
import datetime
import os
import re

from core.engines import RawNNEngine, MCTSEngine
from core.encoding import ChessEncoder
# neuralNetwork import is now primarily handled by engine_utils.py
from core.mcts.mcts import MCTS # Retained as MCTS engine uses it directly

from dotenv import load_dotenv
load_dotenv() # Load .env file early

# Import new utility functions
from engine_utils import (
    load_chess_model, 
    managed_uci_engine, 
    get_stockfish_options,
    get_uci_eval_string,
    get_uci_pv_san,
    STOCKFISH_PATH_ENV, # Use the path from engine_utils
    STOCKFISH_DEFAULT_THINK_TIME_ENV # Use the default think time from engine_utils
)
from utils import create_pgn_game_object, save_pgn_file

# Global constants for think times, can be overridden by function params
# STOCKFISH_THINK_TIME and STOCKFISH_THREADS are now sourced from engine_utils.py if not passed directly

class ChessGameUI:
    def __init__(self, model_state_dict: Dict, num_simulations: int = 400, exploration_weight: float = 1.4, aux_stockfish_for_mcts_eval: Optional[chess.engine.SimpleEngine] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = load_chess_model(model_state_dict, self.device)
        if not self.model:
            raise RuntimeError("Failed to load ChessNN model for UI.")

        self.board = chess.Board()
        encoder_instance = ChessEncoder()
        self.engine = MCTSEngine(
            self.model,
            encoder=encoder_instance,
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            stockfish_engine=aux_stockfish_for_mcts_eval # MCTS can use an external SF for its own evals
        )
        print(f"MCTSEngine initialized for UI (Sims: {num_simulations}, Device: {self.device}).")
        self.move_history = []

    def make_engine_move(self) -> Optional[chess.Move]:
        if self.board.is_game_over(): return None
        # select_move in MCTSEngine might print its own thinking/evals
        best_move, _ = self.engine.select_move(self.board) 
        if best_move:
            print(f"Engine (MCTS) plays: {best_move.uci()}")
            self.board.push(best_move)
            self.move_history.append(best_move)
            return best_move
        print("Engine (MCTS) could not find a move.")
        return None

    def make_player_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
    
    def get_legal_moves(self) -> List[chess.Move]: return list(self.board.legal_moves)
    def is_game_over(self) -> bool: return self.board.is_game_over()
    def get_result(self) -> str: return self.board.result()
    def get_board(self) -> chess.Board: return self.board
    
    def undo_move(self) -> bool:
        if self.move_history: self.board.pop(); self.move_history.pop(); return True
        return False
        
    def reset_game(self): self.board.reset(); self.move_history = []

def play_human_vs_engine(model_state_dict: Dict, num_simulations: int, exploration_weight: float = 1.4, use_stockfish_comparison: bool = False):
    print(f"--- Human vs MCTS Engine (Sims: {num_simulations}) ---")
    stockfish_options = get_stockfish_options() # Default options for comparison

    with managed_uci_engine(STOCKFISH_PATH_ENV if use_stockfish_comparison else None, stockfish_options, "comparison") as sf_comparison_engine:
        try:
            game_ui = ChessGameUI(model_state_dict, num_simulations, exploration_weight, aux_stockfish_for_mcts_eval=sf_comparison_engine)
            print(f"You are White. Engine is Black. MCTS using device: {game_ui.engine.device}")

            while not game_ui.is_game_over():
                print("="*30 + f" Move {game_ui.board.fullmove_number} " + "="*30)
                print(game_ui.board)

                if game_ui.board.turn == chess.WHITE: # Human's turn
                    move_uci = input("Your move (UCI, e.g., e2e4), 'undo', or 'reset': ").strip().lower()
                    if move_uci == 'undo':
                        if game_ui.undo_move(): # Undo player move
                            if game_ui.undo_move(): print("Player and Engine moves undone.") # Undo engine move
                            else: print("Player move undone (was start of game or engine failed).")
                        else: print("Cannot undo further.")
                        continue
                    elif move_uci == 'reset': game_ui.reset_game(); print("Game reset."); continue
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if not game_ui.make_player_move(move): print("Illegal move! Try again.")
                    except ValueError: print("Invalid move format! Use UCI (e.g. e2e4).")
                else: # Engine's turn
                    print("Engine thinking (MCTS)...")
                    if not game_ui.make_engine_move():
                        print("Engine failed to make a move. Game over or error.")
                        break
            
            print("--- Game Over ---"); print(game_ui.board); print(f"Result: {game_ui.get_result()}")

        except RuntimeError as rte: print(f"FATAL ERROR: {rte}"); traceback.print_exc()
        except Exception as e: print(f"An error occurred: {e}"); traceback.print_exc()
    
    input("\\nPress Enter to return to the menu...")

def _play_general_engine_game(
    engine1_player, engine1_name: str, 
    engine2_player, engine2_name: str,
    pgn_event_name: str,
    pgn_save_dir: str,
    pgn_filename_prefix: str,
    pgn_additional_headers: Optional[Dict[str, str]] = None,
    aux_eval_stockfish: Optional[chess.engine.SimpleEngine] = None, # For post-move analysis if needed
    stockfish_eval_think_time: float = STOCKFISH_DEFAULT_THINK_TIME_ENV,
    pause_after_move: bool = False,
    mcts_sims_for_header: Optional[int] = None # Only for PGN header if MCTS is one player
):
    board = chess.Board()
    game_pgn = create_pgn_game_object(engine1_name, engine2_name, pgn_event_name, additional_headers=pgn_additional_headers)
    if mcts_sims_for_header and "MCTSSimulations" not in game_pgn.headers: # Add if not already set
        game_pgn.headers["MCTSSimulations"] = str(mcts_sims_for_header)
    current_pgn_node = game_pgn

    print(f"--- {pgn_event_name} ---")
    print(f"White: {engine1_name} | Black: {engine2_name}")
    if isinstance(engine1_player, MCTSEngine): print(f"({engine1_name} MCTS Sims: {engine1_player.num_simulations})")
    if isinstance(engine2_player, MCTSEngine): print(f"({engine2_name} MCTS Sims: {engine2_player.num_simulations})")

    try:
        while not board.is_game_over():
            print("="*30 + f" Move {board.fullmove_number}{'...' if board.turn == chess.BLACK else ''} " + "="*30)
            print(board)

            current_engine, current_engine_name = (engine1_player, engine1_name) if board.turn == chess.WHITE else (engine2_player, engine2_name)
            
            print(f"{current_engine_name} thinking...")
            best_move = None
            move_pgn_comment = ""
            
            # --- Move selection logic based on engine type ---
            if isinstance(current_engine, (MCTSEngine, RawNNEngine)):
                best_move, move_stats_dict = current_engine.select_move(board) # MCTS/RawNN print their own primary eval
                if best_move:
                    if isinstance(current_engine, RawNNEngine):
                        prob = move_stats_dict.get(best_move, 0.0)
                        move_pgn_comment = f"RawNN Prob: {prob:.4f}"
                        print(f"{current_engine_name} plays: {best_move.uci()} ({move_pgn_comment})")
                    else: # MCTSEngine
                        visits = move_stats_dict.get(best_move, 0)
                        # Q-value/SF-eval might be printed by MCTSEngine's select_move if its internal SF is active
                        move_pgn_comment = f"MCTS Visits: {visits}" 
                        print(f"{current_engine_name} plays: {best_move.uci()} ({move_pgn_comment})")

            elif isinstance(current_engine, chess.engine.SimpleEngine): # External UCI engine (e.g. Stockfish player)
                think_time = getattr(current_engine, 'think_time_override', stockfish_eval_think_time) # Allow override
                limit = chess.engine.Limit(time=think_time)
                result = current_engine.play(board, limit)
                best_move = result.move
                if best_move:
                    score_str = get_uci_eval_string(result.info.get("score"), board.turn)
                    pv_str = get_uci_pv_san(board, result.info.get(chess.engine.INFO_PV))
                    move_pgn_comment = f"ExtUCI {score_str}{pv_str}"
                    print(f"{current_engine_name} plays: {best_move.uci()} ({score_str}{pv_str})")
            else:
                print(f"ERROR: Unknown engine type for {current_engine_name}!")
                break
            
            if not best_move:
                print(f"{current_engine_name} failed to find a move or an error occurred.")
                break

            board.push(best_move)
            current_pgn_node = current_pgn_node.add_variation(best_move)
            current_pgn_node.comment = move_pgn_comment.strip()

            # Optional auxiliary Stockfish analysis (full strength) after the move
            if aux_eval_stockfish and aux_eval_stockfish != getattr(current_engine, 'stockfish_engine', None): # Avoid re-eval if MCTS used same SF
                try:
                    limit = chess.engine.Limit(time=stockfish_eval_think_time)
                    info = aux_eval_stockfish.analyse(board, limit) # Analyse board AFTER move
                    sf_eval_str = get_uci_eval_string(info.get("score"), board.turn) # board.turn is next player's turn
                    print(f"  Aux SF eval after {best_move.uci()}: {sf_eval_str}")
                    current_pgn_node.comment += f"; AuxSF {sf_eval_str}"
                except Exception as sf_eval_e: print(f"Aux SF Error evaluating: {sf_eval_e}")
            
            if pause_after_move: input("Press Enter to continue...")

        print("--- Game Over ---"); print(board); game_result = board.result(); print(f"Result: {game_result}")
        game_pgn.headers["Result"] = game_result
        save_pgn_file(game_pgn, pgn_save_dir, pgn_filename_prefix, engine1_name, engine2_name)
        return game_result, True # True indicates MCTS was white if applicable, adapt as needed for specific callers

    except KeyboardInterrupt: print("\\nGame interrupted by user.")
    except Exception as e: print(f"\\nAn error occurred during {pgn_event_name}: {e}"); traceback.print_exc()
    return board.result() if board else "ERROR", False # Default return for errors/interrupts

def play_engine_vs_engine(state_dict1: Dict, state_dict2: Dict, num_simulations: int, exploration_weight: float = 1.4, use_stockfish_eval: bool = False, pause_after_move: bool = False, model1_name: str = "NN1", model2_name: str = "NN2"):
    print(f"--- MCTS ({model1_name}) vs MCTS ({model2_name}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcts_model1 = load_chess_model(state_dict1, device)
    mcts_model2 = load_chess_model(state_dict2, device)
    if not mcts_model1 or not mcts_model2: return

    stockfish_options = get_stockfish_options()
    with managed_uci_engine(STOCKFISH_PATH_ENV if use_stockfish_eval else None, stockfish_options, "MCTS_aux_eval") as sf_aux_engine:
        encoder = ChessEncoder()
        engine1 = MCTSEngine(mcts_model1, encoder, num_simulations, exploration_weight, stockfish_engine=sf_aux_engine)
        engine2 = MCTSEngine(mcts_model2, encoder, num_simulations, exploration_weight, stockfish_engine=sf_aux_engine)

        _play_general_engine_game(
            engine1, model1_name, engine2, model2_name,
            pgn_event_name="MCTS vs MCTS Match",
            pgn_save_dir="saved_engine_games",
            pgn_filename_prefix="Match_MCTS",
            pgn_additional_headers={"MCTSSimulations": str(num_simulations)},
            aux_eval_stockfish=sf_aux_engine if use_stockfish_eval else None, # Can be same as MCTS's internal one
            pause_after_move=pause_after_move
        )
    input("\\nPress Enter to return to the menu...")


def play_raw_vs_mcts(state_dict: Dict, num_simulations: int, exploration_weight: float = 1.4, raw_plays_white: bool = True, use_stockfish_eval: bool = False, pause_after_move: bool = False, model_name: str = "NN"):
    print(f"--- RawNN vs MCTS ({model_name}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    shared_model = load_chess_model(state_dict, device)
    if not shared_model: return

    stockfish_options = get_stockfish_options()
    with managed_uci_engine(STOCKFISH_PATH_ENV if use_stockfish_eval else None, stockfish_options, "comparison_eval") as sf_aux_engine:
        encoder = ChessEncoder()
        raw_nn_player = RawNNEngine(shared_model, encoder)
        # MCTS can use the aux SF for its internal evaluations if desired
        mcts_player = MCTSEngine(shared_model, encoder, num_simulations, exploration_weight, stockfish_engine=sf_aux_engine)

        engine1, name1, engine2, name2 = (raw_nn_player, f"RawNN_{model_name}", mcts_player, f"MCTS_{model_name}") if raw_plays_white else \
                                         (mcts_player, f"MCTS_{model_name}", raw_nn_player, f"RawNN_{model_name}")
        
        _play_general_engine_game(
            engine1, name1, engine2, name2,
            pgn_event_name="RawNN vs MCTS Comparison",
            pgn_save_dir="saved_comparison_games",
            pgn_filename_prefix=f"Compare_{'RawNN_vs_MCTS' if raw_plays_white else 'MCTS_vs_RawNN'}",
            pgn_additional_headers={"MCTSSimulations": str(num_simulations)}, # Belongs to MCTS player
            aux_eval_stockfish=sf_aux_engine if use_stockfish_eval else None,
            pause_after_move=pause_after_move
        )
    input("\\nPress Enter to return to the menu...")

def play_mcts_vs_external_engine(
    model_state_dict: Dict, num_simulations: int, exploration_weight: float,
    uci_engine_path: str, mcts_plays_white: bool = True,
    use_aux_stockfish_eval: bool = False, pause_after_move: bool = False, model_name: str = "MCTS_NN",
    external_engine_think_time: float = STOCKFISH_DEFAULT_THINK_TIME_ENV # Use default from engine_utils
):
    print(f"--- MCTS ({model_name}) vs External UCI Engine ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcts_nn_model = load_chess_model(model_state_dict, device)
    if not mcts_nn_model: return

    # Options for the auxiliary Stockfish (full strength for eval)
    aux_sf_options = get_stockfish_options() 
    # Options for the external UCI engine (can be anything, not necessarily Stockfish)
    # We don't apply ELO limits or specific thread counts here unless passed in uci_engine_path or its own config
    external_uci_options = {} # External engine might have its own defaults or config file

    with managed_uci_engine(STOCKFISH_PATH_ENV if use_aux_stockfish_eval else None, aux_sf_options, "MCTS_aux_eval") as sf_aux_engine, \
         managed_uci_engine(uci_engine_path, external_uci_options, "external_player") as ext_uci_player:
        
        if not ext_uci_player:
            print(f"FATAL: Could not start external UCI player from {uci_engine_path}.")
            return

        # Allow external engine to have a specific think time set for its play
        # This is a bit of a hack, attaching it to the engine object for _play_general_engine_game
        ext_uci_player.think_time_override = external_engine_think_time 

        encoder = ChessEncoder()
        mcts_player = MCTSEngine(mcts_nn_model, encoder, num_simulations, exploration_weight, stockfish_engine=sf_aux_engine)
        
        ext_engine_id_name = ext_uci_player.id.get("name", os.path.basename(uci_engine_path))

        engine1, name1, engine2, name2 = (mcts_player, f"MCTS_{model_name}", ext_uci_player, ext_engine_id_name) if mcts_plays_white else \
                                         (ext_uci_player, ext_engine_id_name, mcts_player, f"MCTS_{model_name}")

        _play_general_engine_game(
            engine1, name1, engine2, name2,
            pgn_event_name="MCTS vs External UCI Match",
            pgn_save_dir="saved_mcts_vs_uci_games",
            pgn_filename_prefix=f"Game_MCTS_vs_{re.sub(r'[^a-zA-Z0-9_.-]', '', ext_engine_id_name)}",
            pgn_additional_headers={"MCTSSimulations": str(num_simulations), "ExternalEngineThinkTime": str(external_engine_think_time)},
            aux_eval_stockfish=sf_aux_engine if use_aux_stockfish_eval else None,
            pause_after_move=pause_after_move,
            stockfish_eval_think_time=STOCKFISH_DEFAULT_THINK_TIME_ENV, # For the aux SF eval
            mcts_sims_for_header=num_simulations if isinstance(mcts_player, MCTSEngine) else None
        )
    # input("\nPress Enter to return to the menu...") # Often automated in main.py loop for this mode

def play_mcts_vs_stockfish_elo(
    model_state_dict: Dict, num_simulations: int, exploration_weight: float,
    stockfish_elo_limit: int, mcts_plays_white: bool = True,
    use_aux_stockfish_eval: bool = False, # For a full-strength Stockfish doing side evaluations
    pause_after_move: bool = False, model_name: str = "MCTS_NN",
    elo_limited_stockfish_think_time: float = STOCKFISH_DEFAULT_THINK_TIME_ENV # Think time for ELO SF
) -> Tuple[str, bool]:
    print(f"--- MCTS ({model_name}, Sims: {num_simulations}) vs Stockfish ELO {stockfish_elo_limit} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_result_for_caller = "ERROR" # Default

    if not STOCKFISH_PATH_ENV:
        print(f"FATAL ERROR: STOCKFISH_PATH not set. This mode requires Stockfish.")
        return game_result_for_caller, mcts_plays_white

    mcts_nn_model = load_chess_model(model_state_dict, device)
    if not mcts_nn_model: return game_result_for_caller, mcts_plays_white

    # Options for the auxiliary Stockfish (full strength if used)
    aux_sf_options = get_stockfish_options()
    # Options for the ELO-limited Stockfish player (force 1 thread for ELO for consistency unless user specified otherwise for this engine)
    elo_sf_options = get_stockfish_options(elo_limit=stockfish_elo_limit, threads=1) 

    with managed_uci_engine(STOCKFISH_PATH_ENV if use_aux_stockfish_eval else None, aux_sf_options, "aux_eval_full_strength") as aux_eval_sf_engine, \
         managed_uci_engine(STOCKFISH_PATH_ENV, elo_sf_options, f"Stockfish_ELO{stockfish_elo_limit}") as elo_limited_sf_player:

        if not elo_limited_sf_player:
            print(f"FATAL ERROR: Could not initialize ELO-limited Stockfish player.")
            return game_result_for_caller, mcts_plays_white
        
        # Attach think time for the ELO limited player
        elo_limited_sf_player.think_time_override = elo_limited_stockfish_think_time

        encoder = ChessEncoder()
        mcts_player = MCTSEngine(mcts_nn_model, encoder, num_simulations, exploration_weight, stockfish_engine=aux_eval_sf_engine)
        
        sf_player_name = f"Stockfish_ELO{stockfish_elo_limit}"
        engine1, name1, engine2, name2 = (mcts_player, f"MCTS_{model_name}", elo_limited_sf_player, sf_player_name) if mcts_plays_white else \
                                         (elo_limited_sf_player, sf_player_name, mcts_player, f"MCTS_{model_name}")

        pgn_headers = {
            "MCTSSimulations": str(num_simulations),
            "StockfishELOPlayedAs": str(stockfish_elo_limit),
            "StockfishELOThinkTime": str(elo_limited_stockfish_think_time)
        }
        
        game_result, _ = _play_general_engine_game( # The second part of tuple (bool) is not directly used by this caller
            engine1, name1, engine2, name2,
            pgn_event_name=f"MCTS vs Stockfish ELO {stockfish_elo_limit}",
            pgn_save_dir="saved_mcts_vs_stockfish_elo_games",
            pgn_filename_prefix=f"Game_MCTS_vs_SFELO{stockfish_elo_limit}",
            pgn_additional_headers=pgn_headers,
            aux_eval_stockfish=aux_eval_sf_engine if use_aux_stockfish_eval else None,
            stockfish_eval_think_time=STOCKFISH_DEFAULT_THINK_TIME_ENV, # For the aux SF
            pause_after_move=pause_after_move,
            mcts_sims_for_header=num_simulations if isinstance(mcts_player, MCTSEngine) else None
        )
        game_result_for_caller = game_result 
    
    # input("\nPress Enter to return to the menu...") # Typically automated by main.py loop
    return game_result_for_caller, mcts_plays_white