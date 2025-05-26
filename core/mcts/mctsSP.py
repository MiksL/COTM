import chess
import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
import traceback
import random
from core.encoding import ChessEncoder # Keep encoder
# Removed concurrent.futures - no parallel sims within MCTS

class MCTS:
    """ Simplified MCTS using direct model inference. """

    def __init__(self, model: nn.Module, encoder: ChessEncoder, initial_board: chess.Board, simulations: int, exploration_weight: float, gpu_inference_server = None):
        self.model = model
        self.encoder = encoder
        self.gpu_inference_server = gpu_inference_server
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval() # Ensure eval mode

        self.initial_board = initial_board.copy()
        self.simulations = simulations
        self.exploration_weight = exploration_weight # c_puct

        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.P = {} # Stores {move: prior_prob} for expanded states

    def _get_policy_and_value(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """ Expansion phase using direct model call. """
        encoded_np = self.encoder.encode_board(board)
        board_tensor = torch.from_numpy(encoded_np).float().unsqueeze(0).to(self.device)

        policy_logits = None
        value = 0.0
        try:
            with torch.no_grad():
                use_amp = self.device.type == 'cuda'
                # Use torch.cuda.amp.autocast for potential mixed precision speedup
                with torch.amp.autocast(device_type="cuda",enabled=use_amp):
                    policy_logits_tensor, value_tensor = self.gpu_inference_server.submit(encoded_np)
                value = value_tensor.item()
                policy_logits = policy_logits_tensor.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error during direct model inference for {board.fen()}: {e}"); traceback.print_exc()
            return {}, 0.0 # Fallback

        # Process policy
        policy_size = policy_logits.shape[0]
        expected_policy_size = self.encoder.num_possible_moves
        if policy_size != expected_policy_size:
            print(f"FATAL: Direct inference policy size ({policy_size}) != Encoder expected size ({expected_policy_size})!")
            return {}, value # Return value, but empty policy if size mismatch

        legal_moves = list(board.legal_moves)
        if not legal_moves: return {}, value

        policy_map = {}
        logits_for_legal, valid_moves = [], []
        for move in legal_moves:
            try:
                idx = self.encoder.encode_move(move)
                if 0 <= idx < policy_size:
                    logits_for_legal.append(policy_logits[idx])
                    valid_moves.append(move)
            except Exception: pass # Ignore errors for specific moves

        if not valid_moves: return {}, value

        # Softmax normalization
        logits_array = np.array(logits_for_legal, dtype=np.float64)
        probabilities = np.exp(logits_array - np.max(logits_array))
        probabilities /= (np.sum(probabilities) + 1e-9)
        policy_map = {move: prob for move, prob in zip(valid_moves, probabilities)}
        return policy_map, value

    def _select_move_puct(self, board: chess.Board) -> chess.Move | None:
        """ Selects the best child move using PUCT. """
        board_fen = board.fen()
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None

        if board_fen not in self.P:
            # This *shouldn't* happen in the purely sequential run_simulations
            print(f"CRITICAL Error: Node {board_fen} priors not found during selection!")
            return random.choice(legal_moves) # Simple fallback

        n_parent = sum(self.N.get(board_fen + m.uci(), 0) for m in legal_moves)
        sqrt_n_parent = math.sqrt(max(1.0, float(n_parent)))

        best_value = -float('inf')
        best_move = None
        prior_policy_map = self.P.get(board_fen, {})

        for move in legal_moves:
            move_key = board_fen + move.uci()
            q_value = self.Q.get(move_key, 0.0)
            n_move = self.N.get(move_key, 0)
            p_prior = prior_policy_map.get(move, 1e-6)

            puct_score = q_value + self.exploration_weight * p_prior * sqrt_n_parent / (1 + n_move)

            if puct_score > best_value:
                best_value = puct_score
                best_move = move

        if best_move is None: # Fallback
            best_move = random.choice(legal_moves)
        return best_move

    def _get_game_result(self, board: chess.Board) -> float:
        """ Gets terminal game result. """
        if board.is_checkmate(): return -1.0
        elif board.is_stalemate() or \
             board.is_insufficient_material() or \
             board.is_seventyfive_moves() or \
             board.is_fivefold_repetition(): return 0.0
        else: return 0.0 # Should not happen

    def backpropagate(self, path: list[str], value: float):
        """ Backpropagates value up the path. """
        current_value = -value
        for move_key in reversed(path):
            self.N[move_key] += 1
            self.Q[move_key] += (current_value - self.Q[move_key]) / self.N[move_key]
            current_value *= -1

    def _run_one_simulation(self):
        """ Runs a single MCTS simulation. """
        node = self.initial_board.copy()
        path = []
        value = 0.0

        while True:
            node_fen = node.fen()
            if node.is_game_over():
                value = self._get_game_result(node)
                break

            if node_fen not in self.P:
                # --- Expansion ---
                try:
                    policy_map, value = self._get_policy_and_value(node)
                    self.P[node_fen] = policy_map
                except Exception as e:
                    print(f"Error during expansion {node_fen}: {e}"); traceback.print_exc()
                    value = 0.0 # Fallback value on critical error
                break # Backpropagate value after expansion/error

            # --- Selection ---
            best_move = self._select_move_puct(node)
            if best_move is None:
                value = self._get_game_result(node); break

            path.append(node_fen + best_move.uci())
            node.push(best_move)

        self.backpropagate(path, value)

    def run_simulations(self):
        """ Runs all simulations sequentially. """
        if self.simulations <= 0: return
        for _ in range(self.simulations):
            self._run_one_simulation()

    def get_best_move_and_stats(self) -> tuple[chess.Move | None, dict, dict]:
        """ Runs simulations sequentially and returns best move/stats. """
        self.run_simulations() # Always run sequentially now

        root_fen = self.initial_board.fen()
        legal_moves = list(self.initial_board.legal_moves)
        if not legal_moves: return None, {}, {}

        root_visit_stats, root_q_stats = {}, {}
        max_visits, best_move = -1, None

        for move in legal_moves:
            key = root_fen + move.uci()
            visits, q_val = self.N.get(key, 0), self.Q.get(key, 0.0)
            root_visit_stats[move], root_q_stats[move] = visits, q_val
            if visits > max_visits: max_visits, best_move = visits, move

        if best_move is None: best_move = legal_moves[0] # Fallback

        return best_move, root_visit_stats, root_q_stats