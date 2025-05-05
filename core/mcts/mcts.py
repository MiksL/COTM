import chess
import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
import traceback 
import random
from core.encoding import ChessEncoder
from neural_network.neuralNetwork import ChessNN 
from concurrent.futures import ThreadPoolExecutor

class MCTS:
    """
    Monte Carlo Tree Search implementation using PUCT and a Neural Network.
    
    """
    def __init__(self, model: nn.Module, encoder, initial_board: chess.Board, simulations: int, exploration_weight: float):
        """
        Initialize MCTS.

        Args:
            model: The pre-trained PyTorch neural network model (ChessNN instance).
            encoder: An object with `encode_board` and `encode_move` methods (ChessEncoder instance).
            initial_board: The starting chess.Board state for the search.
            simulations: The number of MCTS simulations to run.
            exploration_weight: The exploration constant (c_puct) in the PUCT formula.
        """
        self.model = model
        self.encoder = encoder 
        try:
            self.device = next(model.parameters()).device # Get model's device
        except StopIteration:
             print("Warning: Model has no parameters. Assuming CPU device.")
             self.device = torch.device('cpu')
        self.model.to(self.device) # Ensure model is on correct device
        self.model.eval() # Ensure model is in evaluation mode

        self.initial_board = initial_board.copy()
        self.simulations = simulations
        self.exploration_weight = exploration_weight # c_puct

        # MCTS tree storage
        self.Q = defaultdict(float) # Q(s,a): Stores the mean action value (perspective of player at s)
        self.N = defaultdict(int)   # N(s,a): Stores the visit count for state-action pair
        self.P = {}                 # P(s): Stores {move: prior_prob} for expanded states s (fen key)

    def _get_policy_and_value(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """
        Expansion phase of MCTS.
        Encodes the board, gets NN policy/value, normalizes policy over legal moves.

        Args:
            board: The chess.Board object.

        Returns:
            A tuple containing:
            - policy_map: A dictionary mapping legal `chess.Move` objects to their normalized prior probabilities.
            - value: The estimated value [-1, 1] of the board state from the perspective of the current player.
        """
        # --- Encode the board state using the encoder ---
        encoded_np = self.encoder.encode_board(board) # Get NumPy array [18, 8, 8]
        # Convert to FloatTensor, add batch dimension, send to device
        board_tensor = torch.FloatTensor(encoded_np).unsqueeze(0).to(self.device)

        if board_tensor.shape != (1, 18, 8, 8):
             raise ValueError(f"Encoded board tensor has unexpected shape: {board_tensor.shape}. Expected [1, 18, 8, 8].")

        # --- Send tensor to the NN model and wait for results ---
        with torch.no_grad():
            #  Model returns policy_logits, value
            policy_logits_tensor, value_tensor = self.model(board_tensor)

        # --- Process the NN's returned policy and value ---
        value = value_tensor.item() # NN value estimate [-1, 1]
        policy_logits = policy_logits_tensor.squeeze().cpu().numpy() # Shape (1968,)
        policy_size = policy_logits.shape[0]
        #if policy_size != 1968:
            #print(f"FATAL: NN policy output size is {policy_size}, but expected 1968 based on encoder!")

        legal_moves = list(board.legal_moves)
        if not legal_moves: return {}, value # Return empty policy if no legal moves

        valid_legal_moves = []
        logits_for_legal_moves = []
        for move in legal_moves:
            try:
                # Encode the move using the encoder
                nn_index = self.encoder.encode_move(move) # Get the index (0-1968)
                if 0 <= nn_index < policy_size:
                    valid_legal_moves.append(move)
                    logits_for_legal_moves.append(policy_logits[nn_index])
                else:
                     # Should not happen if encode_move is correct for 4416 output
                     print(f"Error: Index {nn_index} from encode_move is out of bounds (size={policy_size}). Check encode_move implementation.")
            except Exception as e:
                 print(f"Error encoding move {move.uci()} or accessing policy logit: {e}")

        if not valid_legal_moves:
             print("Warning: No valid policy logits found for any legal moves!")

        # Softmax normalization over legal moves only
        logits_array = np.array(logits_for_legal_moves, dtype=np.float64)
        logits_array -= np.max(logits_array) # Improve numerical stability
        exp_logits = np.exp(logits_array)
        probabilities = exp_logits / (np.sum(exp_logits) + 1e-9) 

        policy_map = {move: prob for move, prob in zip(valid_legal_moves, probabilities)}
        return policy_map, value

    def _select_move_puct(self, board: chess.Board) -> chess.Move | None:

        """ Selects the best child move from the current board state using the PUCT formula. """

        board_fen = board.fen()
        legal_moves = list(board.legal_moves)

        if not legal_moves: return None

        # Check if the board state has been expanded
        # If not, we should not be here. 
        if board_fen not in self.P:
             print(f"CRITICAL Error: Node {board_fen} not expanded before selection phase!")

        # Calculate the sum of visits for all legal moves from the parent node
        n_parent = sum(self.N.get(board_fen + m.uci(), 0) for m in legal_moves)

        # Calculate the square root of the number of visits to the parent node
        sqrt_n_parent = math.sqrt(max(1.0, float(n_parent))) # Use max(1,...) avoids sqrt(0)

        # Select the best move using PUCT
        best_value = -float('inf')
        best_move = None
        prior_policy_map = self.P.get(board_fen, {}) # Get stored priors

        # Iterate over all legal moves and calculate PUCT score
        for move in legal_moves:

            move_key = board_fen + move.uci() # State-action key
            q_value = self.Q.get(move_key, 0.0) # Current average value for this action
            n_move = self.N.get(move_key, 0)   # Visits for this action
            p_prior = prior_policy_map.get(move, 1e-6) # Prior probability from NN (use epsilon if missing)

            # PUCT formula: Q(s,a) + c_puct * P(a|s) * sqrt(Sum_b N(s,b)) / (1 + N(s,a))
            # Q is from the perspective of the player choosing move 'a' at state 's'
            puct_score = q_value + self.exploration_weight * p_prior * sqrt_n_parent / (1 + n_move)

            if puct_score > best_value:
                best_value = puct_score
                best_move = move

        # Fallback if something went wrong (e.g., all priors were zero, all Q were -inf)
        if best_move is None and legal_moves:
            print("Warning: No best move selected via PUCT, choosing randomly.")
            best_move = random.choice(legal_moves)

        return best_move

    def _get_game_result(self, board: chess.Board) -> float:
        """
        Get the result of the game from a terminal board state.
        Returns value from the perspective of the player whose turn it IS.
        -1.0 means the current player is checkmated (lost).
        +1.0 would mean the opponent is checkmated (won) - not possible if game ended *on* your turn.
         0.0 for a draw.
        """
        if board.is_checkmate():
            # The game ended, and it's currently player X's turn.
            # This means player Y delivered checkmate on the previous move. Player X lost.
            return -1.0

        elif board.is_stalemate() or \
             board.is_insufficient_material() or \
             board.is_seventyfive_moves() or \
             board.is_fivefold_repetition():
            # Draw conditions
            return 0.0
        else:
            # Should only be called on terminal states
            print(f"Warning: _get_game_result called on non-terminal board: {board.fen()}")
            return 0.0 # Return neutral value if called incorrectly

    def backpropagate(self, path: list[str], value: float):
        """
        Backpropagate the simulation result (value) up the path taken.
        Value is from the perspective of the player at the leaf node.
        Needs negation for parent update.
        """
        # Value estimates the outcome for the player whose turn it is at the *leaf*.
        # When updating Q(s,a) for the parent state 's', the value needs to be
        # from the perspective of the player at 's', which is the opponent of
        # the player at the leaf. So, we negate the value.
        current_value = -value
        for move_key in reversed(path):
            self.N[move_key] += 1
            # Update Q using incremental average: Q_new = Q_old + (value_from_perspective - Q_old) / N
            self.Q[move_key] += (current_value - self.Q[move_key]) / self.N[move_key]
            # Negate value again for the next level up (the perspective flips again)
            current_value *= -1

    def run_simulations(self):
         """ Runs the MCTS simulations from the initial board state. """
         for _ in range(self.simulations):
            node = self.initial_board.copy() # Start simulation from root
            path = [] # Record state-action keys (fen + uci) visited

            # --- Selection Phase ---
            # Traverse the tree using PUCT until a leaf node is reached
            while True:
                node_fen = node.fen()
                if node.is_game_over():
                    # Reached a terminal node during selection
                    value = self._get_game_result(node) # Result from perspective of player at node
                    break # Exit selection loop, proceed to backpropagation

                if node_fen not in self.P:
                     # Reached an unexpanded leaf node
                     # --- Expansion Phase ---
                     try:
                         # Expand the node: get policy priors and value estimate from NN
                         policy_map, value = self._get_policy_and_value(node)
                         if not policy_map and not node.is_game_over():
                              # Handle case where NN predicts poorly or encoding fails but game not over
                              print(f"Warning: Expansion failed for non-terminal node {node_fen}. Treating as draw value.")
                              value = 0.0 # Assign neutral value
                         self.P[node_fen] = policy_map # Store policy priors for this node
                         # Perspective: 'value' is from the NN, representing outcome for player at 'node'
                     except Exception as e:
                          print(f"CRITICAL Error during node expansion {node_fen}: {e}")
                          traceback.print_exc()
                          value = 0.0 # Fallback value if expansion fails catastrophically
                     break # Exit selection loop, proceed to backpropagation
                else:
                    # Node already expanded, select best child using PUCT to continue descent
                    best_move = self._select_move_puct(node)

                    if best_move is None: # Should only happen if game somehow ended after expansion check
                        print(f"Warning: _select returned None for non-terminal node {node_fen}. Treating as terminal.")
                        value = self._get_game_result(node) # Get actual result if possible
                        break # Exit selection loop

                    # Record the state-action key BEFORE pushing the move
                    path.append(node_fen + best_move.uci())
                    node.push(best_move) # Descend the tree

            # --- Backpropagation Phase ---
            # Backpropagate the value obtained from the leaf node (NN eval or terminal result)
            # 'value' is from the perspective of the player AT THE LEAF NODE
            self.backpropagate(path, value)
            
    def run_simulations_parallel(self, num_threads=4):
        """
        Runs MCTS simulations in parallel using multiple threads.
        """
        def run_simulation_subset(simulations_to_run):
            for _ in range(simulations_to_run):
                node = self.initial_board.copy()
                path = []
                while True:
                    node_fen = node.fen()
                    if node.is_game_over():
                        value = self._get_game_result(node)
                        break
                    if node_fen not in self.P:
                        try:
                            policy_map, value = self._get_policy_and_value(node)
                            self.P[node_fen] = policy_map
                        except Exception as e:
                            print(f"Error during expansion: {e}")
                            value = 0.0
                        break
                    else:
                        best_move = self._select_move_puct(node)
                        if best_move is None:
                            value = self._get_game_result(node)
                            break
                        path.append(node_fen + best_move.uci())
                        node.push(best_move)
                self.backpropagate(path, value)

        print(f"Running {self.simulations} MCTS simulations in parallel using {num_threads} threads.")
        # Divide simulations among threads
        simulations_per_thread = self.simulations // num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_simulation_subset, simulations_per_thread) for _ in range(num_threads)]
            for future in futures:
                future.result()  # Wait for all threads to finish

    def get_best_move_and_stats(self) -> tuple[chess.Move | None, dict, dict]:
        """
        Runs simulations and returns the best move, visit counts, and Q-values for root moves.

        Returns:
            tuple: (best_move, root_visit_stats, root_q_stats)
                   best_move: The selected chess.Move based on visit counts.
                   root_visit_stats: Dict mapping legal moves from root to their MCTS visit counts.
                   root_q_stats: Dict mapping legal moves from root to their average Q-values.
        """
        # Perform the MCTS search for the current initial_board state
        try:
            self.run_simulations()
        except NotImplementedError as e:
             print(f"\n--- MCTS Run FAILED (NotImplementedError): {e} ---")
             return None, {}, {} #  return empty dicts
        except Exception as e:
             print(f"\n--- MCTS Run FAILED (Error): {e} ---"); traceback.print_exc()
             return None, {}, {} #  return empty dicts

        # --- Final Move Selection & Stat Collection ---
        root_fen = self.initial_board.fen()
        legal_moves = list(self.initial_board.legal_moves)
        if not legal_moves:
            print("No legal moves from the initial position.")
            return None, {}, {} #  return empty dicts

        root_visit_stats = {}
        root_q_stats = {} #  Dictionary to store Q-values
        max_visits = -1
        best_move = None

        #print("\n--- MCTS Root Node Stats ---")
        #print("Move   | Visits | Avg Value (Q)")
        #print("-------|--------|---------------")
        move_stats_list = []

        # Collect stats for final decision and printing
        for move in legal_moves:
            move_key = root_fen + move.uci()
            visits = self.N.get(move_key, 0)
            avg_value = self.Q.get(move_key, 0.0) # Get Q value
            root_visit_stats[move] = visits # Store visits
            root_q_stats[move] = avg_value #  Store Q value
            move_stats_list.append((move, visits, avg_value))

            if visits > max_visits:
                 max_visits = visits
                 best_move = move

        # Sort stats by visits for printing
        # move_stats_list.sort(key=lambda x: x[1], reverse=True)
        # for move, visits, avg_value in move_stats_list[:15]: # Print top 15
        #     print(f"{str(move):<6} | {visits:<6} | {avg_value:+.4f}")


        # Fallback if no visits recorded
        if best_move is None and legal_moves:
             print("Warning: No move selected after simulations (all visits 0?), choosing first legal move.")
             best_move = legal_moves[0]
             # Assign dummy stats if choosing fallback
             if best_move in root_visit_stats: root_visit_stats[best_move] = max(1, root_visit_stats[best_move])
             else: root_visit_stats[best_move] = 1
             if best_move not in root_q_stats: root_q_stats[best_move] = 0.0


        #print(f"\nSelected best move (by visits): {best_move} ({max_visits} visits)")
        #  Return Q-values as well
        return best_move, root_visit_stats, root_q_stats
