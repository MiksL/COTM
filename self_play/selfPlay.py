import chess
import torch
import numpy as np
import concurrent.futures
import threading
import os
from tqdm import tqdm

from core.mcts.mctsSP import MCTS
import time 
import traceback
from self_play.datasetSP import ChessSPDataset


class SelfPlayChess:
    def __init__(self, model, encoder, num_games=100, temperature=1.0,
                 mcts_simulations=50,
                 exploration_weight=1.4,
                 num_workers=None):

        self.model = model # Keep the model
        self.encoder = encoder
        self.num_games = num_games
        self.temperature = temperature
        self.mcts_simulations = mcts_simulations
        self.exploration_weight = exploration_weight

        if num_workers is None:
            num_workers = max(12, os.cpu_count() or 1)
        self.num_workers = num_workers

        self.lock = threading.Lock()
        self.positions = []
        self.policy_targets = []
        self.values = []
        self.total_positions = 0
        self.total_games = 0
        
        from self_play.gpuInference import GPUInferenceServer

        # After you create your model (initial_model)
        self.gpu_inference_server = GPUInferenceServer(model, batch_size=256, max_wait_ms=9)

    def simulate_game(self):
        """ Simulate game using MCTS with direct inference. """
        mcts_step_timings = []
        board_push_timings = []
        game_over_check_timings = []
        
        game_positions = []
        game_policy_targets = []
        board = chess.Board()
        move_count = 0

        while not board.is_game_over():
            
            # Time board encoding time
            encoded_board = self.encoder.encode_board(board)
            game_positions.append(encoded_board)
            
            # --- Time MCTS step ---
            mcts_step_start_time = time.perf_counter() # Use a different name

            # --- MCTS Search ---
            # *** Pass the model directly to the single MCTS class ***
            mcts_instance = MCTS(
                model=self.model, # Pass model
                encoder=self.encoder,
                initial_board=board,
                simulations=self.mcts_simulations,
                exploration_weight=self.exploration_weight,
                gpu_inference_server=self.gpu_inference_server
            )
            # MCTS now runs simulations sequentially internally
            best_move, root_visit_stats, root_q_stats = mcts_instance.get_best_move_and_stats()

            if best_move is None: break # Exit if MCTS fails

            # --- Create Policy Target (same logic) ---
            policy_target = np.zeros(self.encoder.num_possible_moves, dtype=np.float32)
            total_visits = sum(root_visit_stats.values())
            if total_visits > 0:
                for move, visits in root_visit_stats.items():
                    try:
                        move_idx = self.encoder.encode_move(move)
                        if move_idx != -1:
                            policy_target[move_idx] = visits / total_visits
                    except Exception: pass # Ignore encoding errors
            else: # Handle zero visits (assign uniform probability)
                legal_moves = list(board.legal_moves)
                num_legal = len(legal_moves)
                if num_legal > 0:
                    prob = 1.0 / num_legal
                    for move in legal_moves:
                         try:
                              move_idx = self.encoder.encode_move(move)
                              if move_idx != -1: policy_target[move_idx] = prob
                         except Exception: pass

            game_policy_targets.append(policy_target)

            # --- Select Move to Play (same logic) ---
            move_to_play = best_move # Or add temperature sampling

            board.push(move_to_play)
            move_count += 1

        # --- Game End & Data Storage (same logic) ---
        result = board.result()
        final_value = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}.get(result, 0.0)
        game_values = []
        current_perspective_value = -final_value
        for _ in range(len(game_positions)):
            game_values.append(current_perspective_value)
            current_perspective_value *= -1
        game_values.reverse()

        with self.lock:
            self.positions.extend(game_positions)
            self.policy_targets.extend(game_policy_targets)
            self.values.extend(game_values)
            self.total_positions += len(game_positions)
            self.total_games += 1

        return len(game_positions)

    def run_self_play(self):
        """ Run self-play games (parallel games, sequential MCTS/inference). """
        print(f"Starting self-play ({self.num_games} games, direct inference)...")
        print(f"Using {self.num_workers} worker threads, {self.mcts_simulations} simulations/move.")

        # Reset data
        self.positions, self.policy_targets, self.values = [], [], []
        self.total_positions, self.total_games = 0, 0
        start_time = time.time()

        try:
            # Keep parallel game execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self.simulate_game) for _ in range(self.num_games)}
                with tqdm(total=self.num_games, desc="Simulating Games (Direct MCTS)") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            positions_added = future.result()
                            pbar.update(1)
                            # Update postfix (access shared counters safely)
                            with self.lock:
                                avg_positions = self.total_positions / max(1, self.total_games)
                                elapsed = time.time() - start_time
                                pos_per_sec = self.total_positions / max(1e-6, elapsed)
                            pbar.set_postfix({
                                'games': self.total_games, 'pos': self.total_positions,
                                'avg_len': f'{avg_positions:.1f}', 'pos/s': f'{pos_per_sec:.1f}'
                            })
                        except Exception as e:
                            print(f"\nGame simulation failed: {e}"); traceback.print_exc()
        except Exception as e:
             print(f"\nError during parallel self-play execution: {e}"); traceback.print_exc()
        # Removed: finally block with server shutdown

        # --- Data Conversion & Return (same logic) ---
        positions_np = np.array(self.positions, dtype=np.uint8)
        policy_targets_np = np.array(self.policy_targets, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32).reshape(-1, 1)

        print(f"\nFinished self-play generation.")
        # ... (print stats) ...
        print(f" Data Shapes: Pos {positions_np.shape}, Pol {policy_targets_np.shape}, Val {values_np.shape}")

        # Return dataset object
        return ChessSPDataset(positions_np, policy_targets_np, values_np)