import chess
import chess.engine
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import traceback
from typing import Dict, Tuple, Optional
import numpy as np
import math
from collections import defaultdict
import random
import os

from encoding import ChessEncoder
from neuralNetwork import ChessNN # Assumes nn.Module base
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
# == RawNN Engine Class ==
# ==============================================================================
class RawNNEngine:
    """A simple engine that picks the highest probability legal move from the NN policy head."""
    def __init__(self, model: nn.Module, encoder: ChessEncoder):
        """
        Initializes the Raw NN Engine.

        Args:
            model: The pre-trained PyTorch model instance (ChessNN).
            encoder: The ChessEncoder instance.
        """
        self.model = model
        self.encoder = encoder
        try:
             # Ensure model has parameters before checking device
             if list(model.parameters()):
                 self.device = next(model.parameters()).device
             else:
                 print("Warning (RawNNEngine): Model has no parameters. Assuming CPU.")
                 self.device = torch.device('cpu')
        except StopIteration:
             print("Warning (RawNNEngine): Model has no parameters. Assuming CPU.")
             self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print("RawNNEngine initialized.")

    def select_move(self, board: chess.Board) -> Tuple[chess.Move | None, Dict[chess.Move, float]]:
        """
        Selects the move with the highest raw policy probability.

        Args:
            board: The current chess.Board state.

        Returns:
            Tuple: (best_move, move_probabilities)
                   best_move: The selected chess.Move.
                   move_probabilities: Dictionary mapping legal moves to their raw probabilities.
        """
        print("Raw NN Engine thinking...")
        start_time = time.time()
        raw_nn_move: Optional[chess.Move] = None
        best_prob = -1.0
        move_probabilities = {} # Store probs for return

        if board.is_game_over():
            print("Raw NN: Game already over.")
            return None, {}

        try:
            with torch.no_grad():
                # Encode the board state
                encoded_np = self.encoder.encode_board(board)
                board_tensor = torch.FloatTensor(encoded_np).unsqueeze(0).to(self.device)
                if board_tensor.shape != (1, 18, 8, 8):
                    raise ValueError(f"Raw NN: Encoded board tensor shape mismatch: {board_tensor.shape}")

                # Get policy logits from the model
                policy_logits_tensor, _ = self.model(board_tensor)

                # Apply softmax to get probabilities
                policy_probs = F.softmax(policy_logits_tensor, dim=1).squeeze().cpu().numpy()
                policy_size = policy_probs.shape[0]
                if policy_size != 4416:
                    raise ValueError(f"Raw NN: Policy output size mismatch ({policy_size} vs 4416)")
                
                # Get legal moves
                legal_moves = list(board.legal_moves)

                if legal_moves:
                    for move in legal_moves:
                         try:
                             idx = self.encoder.encode_move(move)
                             if 0 <= idx < policy_size:
                                 prob = policy_probs[idx]
                                 move_probabilities[move] = float(prob) # Store prob
                                 if prob > best_prob:
                                     best_prob = prob
                                     raw_nn_move = move
                             else:
                                 # This case should be rare if encoder is correct
                                 print(f"Warning (Raw NN): Index {idx} for {move.uci()} out of bounds ({policy_size}).")
                                 move_probabilities[move] = 0.0 # Assign 0 prob if index invalid
                         except Exception as enc_err:
                              print(f"Warning (Raw NN): Could not encode/process move {move.uci()}: {enc_err}")
                              move_probabilities[move] = 0.0 # Assign 0 prob on error
                else:
                    print("Raw NN: No legal moves.")

        except Exception as e:
            print(f"ERROR getting raw NN move: {e}")
            traceback.print_exc()
            raw_nn_move=None
            move_probabilities = {}

        elapsed = time.time() - start_time
        if raw_nn_move:
            print(f"Raw NN Chose: {str(raw_nn_move):<6} (Prob: {best_prob:.4f}) | Time: {elapsed:.2f}s")
        else:
            print(f"Raw NN failed to choose a move. | Time: {elapsed:.2f}s")

        # Return move and dict mapping moves to probabilities
        return raw_nn_move, move_probabilities

    # Add a helper for consistency if needed by play_raw_vs_mcts Stockfish eval part
    def _get_stockfish_score_str(self, info: Optional[dict]) -> str:
        """Formats the score from Stockfish analysis info into a string."""
        if info is None or "score" not in info or info["score"] is None: return "SF: N/A"
        try:
            score = info["score"].white();
            if score.is_mate(): mate_in = score.mate(); return f"SF: Mate {mate_in}"
            else: cp = score.score(mate_score=30000); return f"SF: {cp / 100.0:+.2f}cp" if cp is not None else "SF: N/A"
        except Exception as e: print(f"ERROR in _get_stockfish_score_str: {e}"); return "SF: FmtErr"


# ==============================================================================
# == MCTSEngine Class ==
# ==============================================================================
class MCTSEngine:
    """Chess engine using MCTS, optionally comparing moves with Stockfish."""

    def __init__(self, model: nn.Module, encoder: ChessEncoder, num_simulations: int = 400, exploration_weight: float = 1.4, stockfish_engine: Optional[chess.engine.SimpleEngine] = None):
        """
        Initializes the MCTS Chess Engine.

        Args:
            model: The pre-trained PyTorch model instance (ChessNN).
            encoder: The ChessEncoder instance.
            num_simulations: Number of MCTS simulations per move.
            exploration_weight: Exploration constant (c_puct).
            stockfish_engine: An optional initialized python-chess Stockfish engine instance.
        """
        if not hasattr(encoder, 'encode_board') or not hasattr(encoder, 'encode_move'):
             raise TypeError("Provided 'encoder' object must have 'encode_board' and 'encode_move' methods.")

        self.model = model
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.stockfish_engine = stockfish_engine
        try:
             if list(model.parameters()): self.device = next(model.parameters()).device
             else: self.device = torch.device('cpu')
        except StopIteration: self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if self.stockfish_engine:
            print(f"MCTS MCTSEngine initialized with Stockfish engine.")
        else:
            print("MCTS MCTSEngine initialized without Stockfish engine.")

    def _get_stockfish_score_str(self, info: Optional[dict]) -> str:
        """Formats the score from Stockfish analysis info into a string."""
        if info is None or "score" not in info or info["score"] is None:
            return "SF: N/A"
        try:
            score = info["score"].white() # Get score from White's perspective
            if score.is_mate():
                mate_in = score.mate()
                return f"SF: Mate {mate_in}"
            else:
                cp = score.score(mate_score=30000) # Use high mate score
                if cp is None: return "SF: N/A"
                return f"SF: {cp / 100.0:+.2f}cp"
        except Exception as e:
            print(f"ERROR in _get_stockfish_score_str: {e}")
            return "SF: FmtErr" # Formatting Error

    def select_move(self, board: chess.Board) -> Tuple[chess.Move | None, Dict[chess.Move, int]]:
        """
        Finds the best move using MCTS and prints comparison including Stockfish evals on separate lines.

        Returns:
            Tuple: (best_move, move_visit_counts)
        """
        start_time = time.time()
        print("-" * 20) # Separator for clarity
        if board.is_game_over():
            print("Game already over.")
            return None, {}

        # --- 1. Get Raw NN Prediction & Confidence ---
        raw_nn_move: Optional[chess.Move] = None
        raw_nn_confidence = 0.0
        raw_nn_sf_eval_str = "SF: N/A" # Default value

        try:
            with torch.no_grad():
                encoded_np = self.encoder.encode_board(board)
                board_tensor = torch.FloatTensor(encoded_np).unsqueeze(0).to(self.device)
                if board_tensor.shape != (1, 18, 8, 8):
                    raise ValueError("Raw NN: Encoded board tensor shape mismatch")

                policy_logits_tensor, _ = self.model(board_tensor)
                policy_probs = F.softmax(policy_logits_tensor, dim=1).squeeze().cpu().numpy()
                policy_size = policy_probs.shape[0]
                if policy_size != 4416:
                    raise ValueError(f"Raw NN: Policy output size mismatch ({policy_size})")

                legal_moves = list(board.legal_moves)
                best_prob = -1.0
                if legal_moves:
                    for move in legal_moves:
                         try:
                             idx = self.encoder.encode_move(move)
                             if 0 <= idx < policy_size:
                                 prob = policy_probs[idx]
                                 if prob > best_prob:
                                     best_prob = prob
                                     raw_nn_move = move
                         except Exception as enc_err:
                              print(f"Warning (Raw NN): Skip {move.uci()}, encode error: {enc_err}")

                    if raw_nn_move:
                        raw_nn_confidence = best_prob
                    else:
                        print("Raw NN: Could not determine best move among legal options.")
                else:
                    print("Raw NN: No legal moves.")

            # --- Evaluate Raw NN Move with Stockfish (if available) ---
            if self.stockfish_engine and raw_nn_move:
                try:
                    board_copy = board.copy()
                    board_copy.push(raw_nn_move)
                    limit = chess.engine.Limit(time=STOCKFISH_THINK_TIME)
                    info = self.stockfish_engine.analyse(board_copy, limit)
                    raw_nn_sf_eval_str = self._get_stockfish_score_str(info)
                except Exception as sf_err:
                    print(f"ERROR: SF Error evaluating raw NN move: {sf_err}")
                    raw_nn_sf_eval_str = "SF: Err"

            # --- Print Raw NN Suggestion (WITHOUT SF score appended) ---
            if raw_nn_move:
                 print(f"Raw NN Suggestion -> Move: {str(raw_nn_move):<6} | Confidence (Prob): {raw_nn_confidence:.4f}")
                 # --- SEPARATE Print for Stockfish Eval for NN Move ---
                 if "N/A" not in raw_nn_sf_eval_str and "Init" not in raw_nn_sf_eval_str and "Err" not in raw_nn_sf_eval_str:
                     print(f"  Stockfish eval for {str(raw_nn_move):<6}: {raw_nn_sf_eval_str.replace('SF: ','')}")

        except Exception as e:
            print(f"ERROR getting raw NN suggestion: {e}")
            traceback.print_exc()
        print("-" * 10) # Separator

        # --- 2. Run MCTS Search ---
        print(f"Starting MCTS search ({self.num_simulations} sims, c_puct={self.exploration_weight})...")
        mcts_instance = MCTS(
            model=self.model,
            encoder=self.encoder, # Pass encoder
            initial_board=board,
            simulations=self.num_simulations,
            exploration_weight=self.exploration_weight
        )

        mcts_best_move: Optional[chess.Move] = None
        root_node_visits = {}
        root_node_q_values = {}
        mcts_sf_eval_str = "SF: N/A" # Default value
        mcts_confidence_percent = 0.0
        mcts_q_value = 0.0

        try:
            # Unpack Q-values as well
            mcts_best_move, root_node_visits, root_node_q_values = mcts_instance.get_best_move_and_stats() # This prints MCTS stats internally

            # --- 3. Evaluate MCTS Move with Stockfish & Print Comparison ---
            if mcts_best_move:
                 total_mcts_visits = sum(root_node_visits.values())
                 mcts_move_visits = root_node_visits.get(mcts_best_move, 0)
                 mcts_confidence_percent = (mcts_move_visits / (total_mcts_visits + 1e-6)) * 100
                 mcts_q_value = root_node_q_values.get(mcts_best_move, 0.0)

                 # --- Evaluate MCTS Move with Stockfish (if available) ---
                 if self.stockfish_engine:
                     # Reuse eval if same move and previous eval was successful
                     if mcts_best_move == raw_nn_move and "N/A" not in raw_nn_sf_eval_str and "Err" not in raw_nn_sf_eval_str:
                         mcts_sf_eval_str = raw_nn_sf_eval_str
                         # print("DEBUG: (MCTS chose same move as raw NN, reusing SF eval)")
                     else:
                         try:
                             board_copy_mcts = board.copy()
                             board_copy_mcts.push(mcts_best_move)
                             limit = chess.engine.Limit(time=STOCKFISH_THINK_TIME)
                             info_mcts = self.stockfish_engine.analyse(board_copy_mcts, limit)
                             mcts_sf_eval_str = self._get_stockfish_score_str(info_mcts)
                         except Exception as sf_err:
                             print(f"ERROR: SF Error evaluating MCTS move: {sf_err}")
                             mcts_sf_eval_str = "SF: Err"

                 # --- Print MCTS Selection (WITHOUT SF score appended) ---
                 print(f"MCTS Selection    -> Move: {str(mcts_best_move):<6} | Visits: {mcts_move_visits} ({mcts_confidence_percent:.2f}%) | Q-Value: {mcts_q_value:+.4f}")
                 # --- SEPARATE Print for Stockfish Eval for MCTS Move ---
                 if "N/A" not in mcts_sf_eval_str and "Init" not in mcts_sf_eval_str and "Err" not in mcts_sf_eval_str:
                      print(f"  Stockfish eval for {str(mcts_best_move):<6}: {mcts_sf_eval_str.replace('SF: ','')}")

            else:
                 print("MCTS did not select a move.")

        except NotImplementedError as e:
             print(f"\n--- MCTS FAILED (NotImplemented): {e} ---"); traceback.print_exc()
             legal_moves=list(board.legal_moves); mcts_best_move=random.choice(legal_moves) if legal_moves else None; root_node_visits={m:0 for m in legal_moves}; 
             if mcts_best_move: root_node_visits[mcts_best_move]=1; print(f"MCTS falling back to random: {mcts_best_move}")
        except Exception as e:
             print(f"\n--- MCTS FAILED (Error): {e} ---"); traceback.print_exc()
             legal_moves=list(board.legal_moves); mcts_best_move=random.choice(legal_moves) if legal_moves else None; root_node_visits={m:0 for m in legal_moves}; 
             if mcts_best_move: root_node_visits[mcts_best_move]=1; print(f"MCTS falling back to random: {mcts_best_move}")

        elapsed = time.time() - start_time
        print(f"Total selection time: {elapsed:.2f}s")
        print("-" * 20) # Separator

        # Return the move chosen by MCTS and its visit counts
        return mcts_best_move, root_node_visits

    def get_position_evaluation(self, board: chess.Board) -> float:
        """Get the model's direct value evaluation of the current position."""
        if board.is_game_over():
             try:
                 # Need an encoder instance here too
                 mcts_helper = MCTS(self.model, self.encoder, board, 0, 0)
                 return mcts_helper._get_game_result(board)
             except Exception as e:
                 print(f"Error getting terminal eval via MCTS helper: {e}")
                 return 0.0

        with torch.no_grad():
            try:
                encoded_position_np = self.encoder.encode_board(board)
                encoded_position = torch.FloatTensor(encoded_position_np).unsqueeze(0).to(self.device)
                if encoded_position.shape != (1, 18, 8, 8):
                    raise ValueError(f"Eval: Shape mismatch {encoded_position.shape}")

                use_amp = self.device.type == 'cuda'
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    _, value_tensor = self.model(encoded_position)
                return value_tensor.item()
            except NotImplementedError as e:
                print(f"ERROR in eval (NotImplemented): {e}")
                return 0.0
            except Exception as e:
                print(f"Error during direct eval: {e}")
                traceback.print_exc()
                return 0.0
