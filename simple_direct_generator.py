import chess
import torch
import numpy as np
import time
import threading
import os
import h5py
import hdf5plugin
import random
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import select
import concurrent.futures

# Import required modules
try:
    from core.encoding import ChessEncoder
    from neural_network.neuralNetwork import ChessNN
except ImportError:
    print("Required modules not found in path. Please run from project root.")
    sys.exit(1)

class SimpleDirectGenerator:
    PIECE_VALUES = {
        chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
        chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
    }

    def __init__(self, model_path, num_threads=8, max_moves=300, temperature=1.0, temp_threshold=30):
        """Ultra-simple game generator that creates games using direct policy from NN."""
        self.model_path = model_path
        self.num_threads = num_threads
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold  # Apply temperature only to first N moves
        
        # Shared data structures with thread safety
        self.lock = threading.RLock()
        self.all_positions = []
        self.all_moves = []
        self.all_values = []
        self.game_results = []
        self.games_completed = 0
        self.running = True
        self.last_output_time = time.time()
        self.output_interval = 10  # 10 seconds
        
        # Auto-save settings
        self.games_since_last_save = 0
        self.auto_save_interval = 10000  # Save every 10,000 games
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        print(f"Initialized SimpleDirectGenerator with {num_threads} threads")
        print(f"Model: {model_path}")
        print(f"Max moves per game: {max_moves}")
        print(f"Temperature: {temperature} (applied to first {temp_threshold} moves)")
        print(f"Auto-saving every {self.auto_save_interval} games.")
        print(f"Type 'exit' to save and quit")
    
    def select_move_from_policy(self, board, policy_array, temperature=1.0):
        """Select a move based on policy distribution and temperature."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Extract probabilities for legal moves only
        move_probs = {}
        for move in legal_moves:
            move_idx = self.encoder.encode_move(move)
            if 0 <= move_idx < len(policy_array):
                move_probs[move] = policy_array[move_idx]
        
        # If no valid probabilities, return random move
        if not move_probs:
            return random.choice(legal_moves)
        
        # Apply temperature
        if temperature != 1.0:
            probs = np.array(list(move_probs.values()))
            if temperature == 0:  # Deterministic
                best_idx = np.argmax(probs)
                best_move = list(move_probs.keys())[best_idx]
                return best_move
            else:
                # Apply temperature
                probs = probs ** (1.0 / temperature)
                probs = probs / np.sum(probs)  # Re-normalize
                
                # Convert back to dictionary
                move_probs = {move: prob for move, prob in zip(move_probs.keys(), probs)}
        
        # Sample from the probability distribution
        moves = list(move_probs.keys())
        probs = np.array(list(move_probs.values()))
        
        # Handle potential numerical issues
        probs = probs / np.sum(probs)
        
        # Choose a move based on the probability distribution
        try:
            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]
        except:
            # Fallback if there's an issue with sampling
            return random.choice(legal_moves)
    
    def generate_game(self, thread_id, model, encoder):
        """Generate a single game using direct neural network policy."""
        # Set a maximum time limit for a single game
        max_game_time = 300  # 5 minutes
        start_time = time.time()
        
        # Use the passed model and encoder instead of creating new ones
        self.model = model
        self.encoder = encoder
        
        # Local arrays to store data
        positions = []
        moves = []
        values = []
        
        # Start a new game
        board = chess.Board()
        move_count = 0
        position_history = {}  # For tracking repetitions
        halfmove_clock = 0  # Track moves without progress
        
        # print(f"Thread {thread_id}: Starting game loop")
        
        try:
            # Main game loop
            while move_count < self.max_moves:
                # --- 1. Check for game termination conditions first ---
                if time.time() - start_time > max_game_time:
                    break
                if board.is_game_over(claim_draw=True):
                    break
                if self._has_insufficient_material(board):
                    break 
                if not self.running:
                    break 
                if board.halfmove_clock >= 100:
                    break
                
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                # --- 2. Get current board state and turn ---
                encoded_position = self.encoder.encode_board(board)
                current_turn = board.turn

                # --- 3. Determine best_move (neural net policy, temperature, draw disincentives) ---
                input_tensor = torch.FloatTensor(encoded_position).unsqueeze(0)
                if self.model.training:
                    print(f"Thread {thread_id}: WARNING - Model was in training mode, forcing eval mode")
                    self.model.eval()
                with torch.no_grad():
                    policy_output, _ = self.model(input_tensor) # value_output not directly used for move selection here
                policy_array = policy_output.squeeze().cpu().numpy()
                if np.sum(policy_array) > 0:
                    policy_array = policy_array / np.sum(policy_array)
                
                move_probabilities = {}
                for move_obj_loop in legal_moves: # renamed 'move' to 'move_obj_loop' to avoid clash
                    move_idx = self.encoder.encode_move(move_obj_loop)
                    if 0 <= move_idx < len(policy_array):
                        prob = policy_array[move_idx]
                        
                        # Capture Bonus for "Trading Up"
                        if board.is_capture(move_obj_loop):
                            attacker_piece_type = board.piece_type_at(move_obj_loop.from_square)
                            attacker_value = self.PIECE_VALUES.get(attacker_piece_type, 0.0)

                            captured_piece_value = 0.0
                            if board.is_en_passant(move_obj_loop):
                                captured_piece_value = self.PIECE_VALUES.get(chess.PAWN, 0.0) # En passant always captures a pawn
                            else:
                                victim_piece_type = board.piece_type_at(move_obj_loop.to_square)
                                if victim_piece_type is not None: # Should be true for non-EP captures
                                    captured_piece_value = self.PIECE_VALUES.get(victim_piece_type, 0.0)
                            
                            if captured_piece_value > attacker_value:
                                #prob *= 1.1  # Apply bonus if captured piece value > attacker piece value
                                pass

                        # Bonus for checks (applied to the potentially capture-boosted probability)
                        board_copy = board.copy() # Create a copy to test the move
                        board_copy.push(move_obj_loop)
                        if board_copy.is_check():
                            #prob *= 1.2  # User-defined factor for checks
                            pass
                        
                        # ADDED: Bonus for Checkmate-in-1
                        # This uses the same board_copy which has move_obj_loop already applied
                        if board_copy.is_checkmate():
                            prob += 0.09
                            prob *= 100.0  # Strong bonus for checkmate
                            # Optionally, print a diagnostic for this
                            print(f"Thread {thread_id}: Mate-in-1 detected with move {move_obj_loop.uci()}! Applying strong bonus.")
                        
                        # Check original board state *before* move_obj_loop is made by board_copy.push()
                        moving_piece_type = board.piece_type_at(move_obj_loop.from_square)
                        if moving_piece_type == chess.PAWN:
                            # board_copy has move_obj_loop already pushed.
                            # So, move_obj_loop.promotion is the correct way to check if *this* move is a promotion.
                            if move_obj_loop.promotion is not None:
                                if move_obj_loop.promotion == chess.QUEEN:
                                    prob *= 1.2  # Strong bonus for queen promotion
                                elif move_obj_loop.promotion == chess.KNIGHT:
                                    # Check if the knight promotion leads to check
                                    if board_copy.is_check():
                                        #prob *= 1.3  # Keep high bonus for knight promotions that give check
                                        pass
                                    else:
                                        #prob *= 1.1  # Lower bonus for non-check knight promotions
                                        pass
                                else:
                                    #prob *= 1.0  # Minimal bonus for rook/bishop promotions
                                    pass
                            else:
                                # If not an immediate promotion, check if it's a push to 7th/2nd rank
                                to_rank = chess.square_rank(move_obj_loop.to_square) # 0-7 (rank 1 to 8)
                                is_white_turn_for_pawn = (board.turn == chess.WHITE) # Turn of the original board state
                                
                                if (is_white_turn_for_pawn and to_rank == 6) or \
                                   (not is_white_turn_for_pawn and to_rank == 1):
                                    # Pawn pushing to 7th rank (White) or 2nd rank (Black)
                                    #prob *= 1.1  # Moderate bonus for getting close (e.g., 1.8x)
                                    pass
                        
                        # Penalty for repetitive positions (using FEN strings)
                        # board_copy already has move_obj_loop pushed to it.
                        rep_fen_eval = board_copy.board_fen() + (" w " if board_copy.turn == chess.WHITE else " b ")

                        # Output for 3-fold repetition check during move consideration
                        current_history_count_eval = position_history.get(rep_fen_eval, 0)
                        if current_history_count_eval >= 2:
                            print(f"Thread {thread_id}: CHECK - Move {move_obj_loop.uci()} would lead to 3-fold repetition. (FEN: {rep_fen_eval} already seen {current_history_count_eval} times in history).")

                        # Refined penalty logic using FEN count
                        if current_history_count_eval >= 2: # If played, this move leads to the 3rd (or more) occurrence
                            prob *= 0.01  # Extremely high penalty for a move that would draw by 3-fold rep.
                        elif current_history_count_eval == 1: # If played, this move leads to the 2nd occurrence
                            prob *= 0.2   # Moderate penalty for repeating once
                        # No penalty if current_history_count_eval is 0
                            
                        # Store adjusted probability
                        move_probabilities[move_obj_loop] = prob
                
                best_move = None
                if move_probabilities:
                    total_prob = sum(move_probabilities.values())
                    if total_prob > 0:
                        for m_key in move_probabilities: move_probabilities[m_key] /= total_prob # Renamed 'move' to 'm_key'
                    current_temp = self.temperature if move_count < self.temp_threshold else 0.1
                    if current_temp == 0:
                        best_move = max(move_probabilities.items(), key=lambda x: x[1])[0]
                    else:
                        temp_moves = list(move_probabilities.keys())
                        temp_probs = np.array(list(move_probabilities.values()))
                        temp_probs = temp_probs ** (1.0 / current_temp)
                        temp_probs = temp_probs / np.sum(temp_probs)
                        if np.any(temp_probs < 0):
                            temp_probs = np.maximum(temp_probs, 0)
                            if np.sum(temp_probs) > 0: temp_probs = temp_probs / np.sum(temp_probs)
                            else: temp_probs = np.ones(len(temp_moves)) / len(temp_moves)
                        try:
                            best_move_idx = np.random.choice(len(temp_moves), p=temp_probs)
                            best_move = temp_moves[best_move_idx]
                        except ValueError as e:
                            print(f"Thread {thread_id}: Sampling error: {e}. Picking random from move_probabilities.")
                            if temp_moves: best_move = random.choice(temp_moves)
                            else: best_move = random.choice(legal_moves) # Absolute fallback
                else: # Fallback if move_probabilities is empty (e.g. no legal moves had policy output)
                    current_temp = self.temperature if move_count < self.temp_threshold else 0.1
                    best_move = self.select_move_from_policy(board, policy_array, current_temp) # Original fallback

                if best_move is None: # If still no best_move (e.g. select_move_from_policy failed)
                    if legal_moves: best_move = random.choice(legal_moves) # Final fallback
                    else: break # No legal moves, cannot continue
                
                # Check if promotion is done
                if best_move.promotion is not None:
                    promoted_piece_name = chess.piece_name(best_move.promotion).upper()
                    print(f"Thread {thread_id}: PROMOTION - Move {best_move.uci()} promotes to a {promoted_piece_name} in current game (approx game {self.games_completed + 1}, move {move_count + 1}).")
                
                # --- 4. Encode the chosen best_move ---
                current_encoded_move = -1
                try:
                    current_encoded_move = self.encoder.encode_move(best_move)
                    if not isinstance(current_encoded_move, (int, np.integer)) or current_encoded_move < 0:
                        print(f"Thread {thread_id}: Invalid encoded move ({current_encoded_move}) for {best_move.uci()}. Breaking game.")
                        break
                except Exception as enc_ex:
                    print(f"Thread {thread_id}: Error encoding move {best_move.uci()}: {enc_ex}. Breaking game.")
                    break

                # --- 5. If all successful, append to local lists and apply move --- 
                positions.append((encoded_position, current_turn))
                moves.append(current_encoded_move)
                
                board.push(best_move)
                move_count += 1
                
                # Update repetition history using FEN strings
                rep_fen_actual = board.board_fen() + (" w " if board.turn == chess.WHITE else " b ")
                position_history[rep_fen_actual] = position_history.get(rep_fen_actual, 0) + 1
                if position_history.get(rep_fen_actual, 0) >= 3:
                    # Optional: For debugging, you could print board.fen() here if needed
                    # print(f"Thread {thread_id}: Breaking game (3-fold repetition by FEN) after {move_count} moves. FEN: {rep_fen_actual}")
                    break
        
        except Exception as e:
            print(f"Thread {thread_id}: Error during game generation: {e}")
            import traceback
            traceback.print_exc()
        
        # Force completion if needed
        if not board.is_game_over(claim_draw=True) and self._has_insufficient_material(board):
            # print(f"Thread {thread_id}: Forcing draw due to insufficient material")
            # Force the result to draw
            result_str = "1/2-1/2"
        elif not board.is_game_over(claim_draw=True):
            # print(f"Thread {thread_id}: Game not naturally concluded, marking as draw")
            # Output FEN
            print("Board state: ", board.board_fen())
            result_str = "1/2-1/2"
        else:
            result_str = board.result(claim_draw=True)
        
        # Double-check result
        if result_str == "*":
            print(f"Thread {thread_id}: Warning - Got '*' result, forcing to draw")
            print(f"Thread {thread_id}: Board state: {board.fen()}")
            print(f"Thread {thread_id}: White pieces: {[p for p in board.piece_map().values() if p.color == chess.WHITE]}")
            print(f"Thread {thread_id}: Black pieces: {[p for p in board.piece_map().values() if p.color == chess.BLACK]}")
            result_str = "1/2-1/2"
        
        final_value_white = 1.0 if result_str == "1-0" else (-1.0 if result_str == "0-1" else 0.0)
        
        # print(f"Thread {thread_id}: Game ended with result {result_str}, {move_count} moves") # Verbose, keep commented
        
        # Only continue if we have positions to process
        if len(positions) > 0:
            # At this point, len(positions) and len(moves) should be identical due to the new main loop structure.
            # Assign values based on game result.
            # The local 'positions' list contains (encoded_board_state, turn_for_that_state) tuples.
            # The local 'moves' list contains the encoded move taken from that state.
            # The local 'values' list will store the game outcome from the perspective of the player whose turn it was.

            for i in range(len(positions)):
                _, turn_for_this_state = positions[i] # Get the turn from the stored tuple
                is_white_to_move_for_this_state = (turn_for_this_state == chess.WHITE)
                perspective_value = final_value_white if is_white_to_move_for_this_state else -final_value_white
                values.append(perspective_value)
            
            # Data for global extend
            # Extract only the board states from the (board_state, turn) tuples for self.all_positions
            plain_positions = [p_tuple[0] for p_tuple in positions]
            current_moves = moves
            current_values = values
            
            # Store results in the global arrays with lock protection
            with self.lock:
                # CRITICAL CHECK before extending global lists
                if not (len(plain_positions) == len(current_moves) == len(current_values)):
                    print(f"Thread {thread_id}: FATAL MISMATCH before global extend! This should not happen.")
                    print(f"  plain_positions: {len(plain_positions)}, current_moves: {len(current_moves)}, current_values: {len(current_values)}")
                    # Decide on a recovery strategy or raise an error if this occurs
                    # For now, we will skip extending to prevent corrupting global data if this unlikely event occurs
                else:
                    self.all_positions.extend(plain_positions)
                    self.all_moves.extend(current_moves)
                    self.all_values.extend(current_values)
                    self.game_results.append(final_value_white)
                    self.games_completed += 1
                    self.games_since_last_save += 1
                
                # Print progress update (moved inside lock to ensure games_completed is up-to-date for print_stats)
                current_time = time.time()
                if current_time - self.last_output_time >= self.output_interval:
                    self.print_stats() # Relies on self.lock
                    self.last_output_time = current_time
                
                # Auto-save logic (moved inside lock to ensure games_completed is up-to-date for save trigger)
                if self.running and self.games_since_last_save >= self.auto_save_interval:
                    print(f"\n--- Auto-saving data ({self.games_completed} total games, {self.games_since_last_save} since last save) ---")
                    self.save_to_hdf5() # Relies on self.lock
                    self.games_since_last_save = 0 # Reset counter
                    print(f"--- Auto-save complete ---")
        else:
            # This case means the game loop broke before any valid moves were made, or positions list was empty.
            # print(f"Thread {thread_id}: No positions generated for this game, discarding.") # Can be noisy
            pass
        
        game_duration = time.time() - start_time
        print(f"Thread {thread_id} completed game with {move_count} moves in {game_duration:.1f}s, result: {result_str}")
        return len(positions)
    
    def print_stats(self):
        """Print current statistics."""
        with self.lock:
            pos_count = len(self.all_positions)
            games_done = len(self.game_results)
            
            print(f"\n--- Status Update ---")
            print(f"Games completed: {games_done}")
            print(f"Total positions: {pos_count}")
            
            # Print result distribution if any games are done
            if games_done > 0:
                # Safe calculation in case of division issues
                white_wins = sum(1 for result in self.game_results if result > 0.5)
                black_wins = sum(1 for result in self.game_results if result < -0.5)
                # games_done should be equal to len(self.game_results)
                draws = len(self.game_results) - white_wins - black_wins 
                print(f"Results - White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
                # Calculate draw percentage
                draw_percentage = (draws / len(self.game_results)) * 100 if len(self.game_results) > 0 else 0
                print(f"Draw rate: {draw_percentage:.1f}%")
    
    def save_to_hdf5(self):
        """Save all collected data to HDF5 file."""
        with self.lock:
            if not self.all_positions:
                print("No positions to save")
                return
            
            # Create local copies of the arrays to avoid thread interference
            positions = self.all_positions.copy()
            moves = self.all_moves.copy()
            values = self.all_values.copy()
            games_count = self.games_completed
            
        # Continue processing outside the lock to minimize lock time
        print(f"Processing {len(positions)} positions from {games_count} games")
        
        # Create a single encoder instance for all conversions - only create it once
        encoder = None
        non_int_count = 0
        
        # Check and fix non-integer moves in a single pass
        for i in range(len(moves)):
            if not isinstance(moves[i], (int, np.integer)):
                # Create encoder only once when needed
                if encoder is None:
                    encoder = ChessEncoder()
                
                non_int_count += 1
                # If it's a Move object, try to encode it
                try:
                    if hasattr(moves[i], 'uci'):
                        # Only print the first few conversion messages
                        if non_int_count <= 5:
                            print(f"Converting move {moves[i]} to integer")
                        elif non_int_count == 6:
                            print("... more moves being converted (messages suppressed)")
                            
                        moves[i] = encoder.encode_move(moves[i])
                    else:
                        # If it's neither an integer nor a Move object, use a placeholder
                        print(f"Found invalid move type: {type(moves[i])}, using 0")
                        moves[i] = 0
                except:
                    # If conversion fails, use placeholder
                    print(f"Failed to convert move, using 0")
                    moves[i] = 0
        
        # Convert to numpy arrays all at once
        positions_np = np.array(positions, dtype=np.uint8)
        moves_np = np.array(moves, dtype=np.int16)
        values_np = np.array(values, dtype=np.float32).reshape(-1, 1)

        # Check for data length mismatches
        if not (positions_np.shape[0] == moves_np.shape[0] == values_np.shape[0]):
            print(f"\nWARNING: Data length mismatch detected during save!")
            print(f"  Positions: {positions_np.shape[0]}")
            print(f"  Moves:     {moves_np.shape[0]}")
            print(f"  Values:    {values_np.shape[0]}")
            print(f"  This indicates an issue in data collection. Saving with potentially misaligned data or truncating might be necessary if error persists.")
            # Consider whether to truncate to the minimum length or handle differently
            # For now, we'll proceed, but the chunking fix below should handle HDF5 part if lists are different lengths.
        
        # Set up compression
        compression_opts = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
        
        # Create output filename with timestamp
        timestamp = int(time.time())
        output_path = f"output/direct_games_{timestamp}.hdf5"
        
        # Calculate chunk sizes for each dataset independently
        # Ensure positions_np.shape[0] > 0 due to early exit if self.all_positions is empty
        positions_chunk_dim0 = min(2048, positions_np.shape[0]) if positions_np.shape[0] > 0 else None
        positions_chunks = (positions_chunk_dim0, 18, 8, 8) if positions_chunk_dim0 is not None else None

        moves_chunk_dim0 = min(2048, moves_np.shape[0]) if moves_np.shape[0] > 0 else None
        moves_chunks = (moves_chunk_dim0,) if moves_chunk_dim0 is not None else None

        values_chunk_dim0 = min(2048, values_np.shape[0]) if values_np.shape[0] > 0 else None
        values_chunks = (values_chunk_dim0, 1) if values_chunk_dim0 is not None else None
        
        # Save to HDF5
        print(f"Saving data: {positions_np.shape[0]} positions, {moves_np.shape[0]} moves, {values_np.shape[0]} values to HDF5")
        with h5py.File(output_path, 'w') as f_out:
            # Create datasets with chunks and compression
            pos_dset = f_out.create_dataset('positions', data=positions_np, 
                                           chunks=positions_chunks, **compression_opts)
            mov_dset = f_out.create_dataset('moves', data=moves_np,
                                           chunks=moves_chunks, **compression_opts)
            val_dset = f_out.create_dataset('values', data=values_np,
                                           chunks=values_chunks, **compression_opts)
            
            # Add metadata
            f_out.attrs['creation_timestamp'] = time.time()
            f_out.attrs['source'] = 'direct_policy_play'
            f_out.attrs['games_count'] = games_count
            f_out.attrs['model_path'] = self.model_path # Keep record of the model used
        
        # Model saving is removed as it's not being trained by this generator.
        # The model_path attribute in HDF5 points to the original model used.
        
        print(f"Data saved to {output_path}")
        return output_path
    
    def run(self):
        """Run game generation in parallel threads."""
        # Start the command input thread
        command_thread = threading.Thread(target=self._handle_commands)
        command_thread.daemon = True
        command_thread.start()
        
        print(f"Starting {self.num_threads} threads for game generation...")
        print("Type 'status' to see progress, 'save' to save current data, or 'exit' to quit")
        
        try:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit tasks for each thread
                futures = []
                for thread_id in range(self.num_threads):
                    future = executor.submit(self._thread_worker, thread_id)
                    futures.append(future)
                
                # Wait for user to stop the process manually
                # The threads will keep running until self.running is set to False
                while self.running:
                    # Check if any threads have failed
                    for future in list(futures):
                        if future.done():
                            try:
                                # Get the result and print
                                positions = future.result()
                                # print(f"Thread completed with {positions} positions")
                                
                                # Remove from our tracking list
                                futures.remove(future)
                                
                                # If we're still running, restart this thread
                                if self.running:
                                    thread_id = len(futures)
                                    new_future = executor.submit(self._thread_worker, thread_id)
                                    futures.append(new_future)
                                    # print(f"Restarted thread with ID {thread_id}")
                            except Exception as e:
                                print(f"Thread failed with error: {e}")
                                import traceback
                                traceback.print_exc()
                                
                                # Remove from our tracking list
                                futures.remove(future)
                                
                                # If we're still running, restart this thread
                                if self.running:
                                    thread_id = len(futures)
                                    new_future = executor.submit(self._thread_worker, thread_id)
                                    futures.append(new_future)
                                    # print(f"Restarted thread with ID {thread_id} after failure")
                    
                    # Sleep to avoid using too much CPU
                    time.sleep(1)
                
                # User has stopped the process, cancel remaining threads
                for future in futures:
                    future.cancel()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user, saving before exit...")
        except Exception as e:
            print(f"Error in main thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always save before exiting
            self.running = False
            self.save_to_hdf5()
            
        print("All threads completed")
    
    def _thread_worker(self, thread_id):
        """Worker function that generates multiple games until stopped"""
        total_positions = 0
        games_generated = 0
        
        # Initialize model and encoder once per thread
        model = ChessNN()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()  # Set to eval mode initially
        
        # Create a single encoder per thread - used for all games
        encoder = ChessEncoder()
        
        # Keep generating games until manually stopped
        while self.running:
            try:
                # Set a timeout for generating each game
                start_time = time.time()
                # Pass the already created encoder to avoid rebuilding it
                positions = self.generate_game(thread_id, model, encoder)
                game_time = time.time() - start_time
                
                if positions > 0:
                    total_positions += positions
                    games_generated += 1
                    #print(f"Thread {thread_id} generated {games_generated} valid games so far")
                # else:
                    # print(f"Thread {thread_id} generated an empty game, not counting")
                
                # Add a small delay between games
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Thread {thread_id} error in game generation: {e}")
                import traceback
                traceback.print_exc()
                # Sleep a bit before retrying
                time.sleep(1)
        
        return total_positions
    
    def _handle_commands(self):
        """Handle user commands during execution."""
        print("Command handler active. Type 'status', 'save', or 'exit'")
        
        while self.running:
            try:
                # Simple blocking input instead of select.select
                command = input().strip().lower()
                
                if command == "status":
                    self.print_stats()
                    # Ensure output is flushed
                    sys.stdout.flush()
                elif command == "save":
                    output_path = self.save_to_hdf5()
                    sys.stdout.flush()
                elif command == "exit" or command == "quit":
                    print("Saving and exiting...")
                    output_path = self.save_to_hdf5()
                    self.running = False
                    break
                else:
                    print("Unknown command. Available: status, save, exit")
                    sys.stdout.flush()
                
            except Exception as e:
                print(f"Error in command handling: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Sleep a bit longer after an error
    
    def _has_insufficient_material(self, board):
        """Check if the position has insufficient material for checkmate."""
        # Count pieces by type for each color
        white_pieces = [p for p in board.piece_map().values() if p.color == chess.WHITE]
        black_pieces = [p for p in board.piece_map().values() if p.color == chess.BLACK]
        
        # King vs King
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True
        
        # King and Knight/Bishop vs King
        if (len(white_pieces) == 2 and len(black_pieces) == 1 and 
            any(p.piece_type in [chess.KNIGHT, chess.BISHOP] for p in white_pieces)):
            return True
            
        if (len(black_pieces) == 2 and len(white_pieces) == 1 and 
            any(p.piece_type in [chess.KNIGHT, chess.BISHOP] for p in black_pieces)):
            return True
            
        # King and 2 Knights vs King (usually drawn)
        if (len(white_pieces) == 3 and len(black_pieces) == 1 and 
            sum(1 for p in white_pieces if p.piece_type == chess.KNIGHT) == 2):
            return True
            
        if (len(black_pieces) == 3 and len(white_pieces) == 1 and 
            sum(1 for p in black_pieces if p.piece_type == chess.KNIGHT) == 2):
            return True
        
        # Check for king and bishop vs king and bishop with same colored bishops
        if len(white_pieces) == 2 and len(black_pieces) == 2:
            white_bishops = [p for p in white_pieces if p.piece_type == chess.BISHOP]
            black_bishops = [p for p in black_pieces if p.piece_type == chess.BISHOP]
            if len(white_bishops) == 1 and len(black_bishops) == 1:
                # Check if bishops are on same colored squares
                white_bishop_square = None
                black_bishop_square = None
                for square, piece in board.piece_map().items():
                    if piece.piece_type == chess.BISHOP:
                        if piece.color == chess.WHITE:
                            white_bishop_square = square
                        else:
                            black_bishop_square = square
                
                if white_bishop_square is not None and black_bishop_square is not None:
                    white_square_color = (white_bishop_square // 8 + white_bishop_square % 8) % 2
                    black_square_color = (black_bishop_square // 8 + black_bishop_square % 8) % 2
                    if white_square_color == black_square_color:
                        return True
        
        return False

def main():
    # Fixed configuration without command line arguments for better QoL
    # Edit these values directly in the code as needed
    model_path = "models/selfPlay-v7-ep02-acc01645.pth"  # Path to model file (.pth)
    num_threads = 15                                      # Number of threads
    max_moves = 100                                       # Maximum moves per game  
    temperature = 1.0                                     # Temperature for move selection
    temp_threshold = 80                                   # Temperature threshold
    
    # Set up draw disincentives in the code:
    # - Captures get 20% probability boost
    # - Checks get 15% probability boost
    # - Repetitive positions get 30% penalty
    # - Uses temperature=0.1 after temp_threshold (instead of 0) to avoid deterministic draws
    
    print(f"Starting game generator with {num_threads} threads")
    print(f"Using {model_path}")
    print(f"Moves: max {max_moves}, temp {temperature} until move {temp_threshold}, then 0.1")
    print(f"Draw avoidance enabled: boosting captures/checks, penalizing repetitions")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please adjust the model_path in simple_direct_generator.py")
        return 1
    
    generator = SimpleDirectGenerator(
        model_path=model_path,
        num_threads=num_threads,
        max_moves=max_moves,
        temperature=temperature,
        temp_threshold=temp_threshold
    )
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving before exit...")
        generator.running = False
        generator.save_to_hdf5()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 