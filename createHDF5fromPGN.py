import chess
import chess.pgn
import numpy as np
import os
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
import gc
import re
import math
import time
from dotenv import load_dotenv
import sys
import traceback
import chess.engine
import h5py
import hdf5plugin

try:
    from encoding import ChessEncoder
except ImportError:
    print("Error: encoding.py not found. Make sure it's in the same directory.")
    sys.exit(1)

# --- HDF Configuration ---
POS_DTYPE = np.uint8
MOVE_DTYPE = np.int16
VALUE_DTYPE = np.float32


SAMPLES_PER_CHUNK = 2048
POS_CHUNK = (SAMPLES_PER_CHUNK, 18, 8, 8)
MOVE_CHUNK = (SAMPLES_PER_CHUNK,)
VALUE_CHUNK = (SAMPLES_PER_CHUNK, 1)

# Blosc compression settings
COMPRESSION_OPTS = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE)

# --- Helper Functions (parse_elo, get_result_value, scale_score, parse_stockfish_eval) ---
def parse_elo(elo_str):
    if isinstance(elo_str, str) and elo_str.isdigit():
        return int(elo_str)
    return 0

def get_result_value(result_str):
    if result_str == '1-0': return 1.0
    elif result_str == '0-1': return -1.0
    elif result_str == '1/2-1/2': return 0.0
    else: return 0.0 # Unknown result = draw

def scale_score(score_obj, turn):
    """Scales a chess.engine.Score object using tanh."""
    pov_score = score_obj.pov(turn)
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        scaled_val = 10.0 if mate_in > 0 else -10.0 # Large value before tanh representing mate
    else:
        cp_score = pov_score.score()
        if cp_score is None: return 0.0
        # Scale centipawns by divisor
        scaled_val = cp_score / DIVISOR
    # Return value scaled to [-1, 1] with tanh    
    return math.tanh(scaled_val)


def parse_stockfish_eval(comment, turn_for_scaling):
    """Parses PGN comment AND applies tanh scaling."""
    if not isinstance(comment, str): return None
    match = re.search(r"\[%eval\s+([#\-]?\d+\.?\d*)]", comment) # regex search for PGN eval
    if not match: return None
    eval_str = match.group(1)
    try:
        if eval_str.startswith('#'):
            mate_score = int(eval_str[1:])
            # Assume PGN eval is from White POV
            score_obj = chess.engine.PovScore(chess.engine.Mate(mate_score), chess.WHITE)
        else:
            # Evaluation is pawn units
            pawn_score = float(eval_str)
            cp_score = int(pawn_score * 100)
            # Assume PGN eval is from White POV
            score_obj = chess.engine.PovScore(chess.engine.Cp(cp_score), chess.WHITE)

        # Scale based on whose turn it actually is
        return scale_score(score_obj, turn_for_scaling)
    except ValueError:
        return None


# --- Worker Function (Processes PGN batch & Returns NumPy arrays) ---
def _process_game_batch_hdf5_ready(batch_data):
    # Unpack arguments
    game_batch, elo_floor, use_stockfish_eval, encoder, stockfish_path, sf_depth, sf_threads = batch_data

    # Lists to hold results before converting to NumPy arrays
    positions_list = []
    moves_list = []
    values_list = []

    games_processed_in_batch = 0
    engine = None

    try:
        # --- Initialize Stockfish Engine ---
        if use_stockfish_eval and stockfish_path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                engine.configure({"Threads": sf_threads})
            except Exception as e:
                print(f"Worker Error: Failed to initialize Stockfish: {e}")
                engine = None

        # --- Process games in the batch ---
        for game_idx, game in enumerate(game_batch):
            # 1. ELO filter
            white_elo = parse_elo(game.headers.get("WhiteElo", "0"))
            black_elo = parse_elo(game.headers.get("BlackElo", "0"))
            if white_elo < elo_floor or black_elo < elo_floor:
                continue

            # 2. Process game
            result_str = game.headers.get("Result", "*")
            game_outcome_value = get_result_value(result_str)
            board = game.board()
            game_yielded_samples = False

            # Process game moves
            node = game.root()
            while node.variations:
                next_node = node.variations[0]
                move = next_node.move

                # --- Determine position value ---
                current_comment = node.comment
                final_value = None
                current_turn = board.turn # Whose turn is it *before* the move

                if use_stockfish_eval:
                    # Try PGN eval first
                    final_value = parse_stockfish_eval(current_comment, current_turn)

                    # If no PGN eval and engine exists, eval with Stockfish
                    if final_value is None and engine:
                        try:
                            limit = chess.engine.Limit(depth=sf_depth)
                            info = engine.analyse(board, limit)
                            score_obj = info.get("score")
                            if score_obj:
                                final_value = scale_score(score_obj, current_turn)
                            else: # Fallback for missing score
                                final_value = game_outcome_value if current_turn == chess.WHITE else -game_outcome_value
                        except Exception: # Fallback on error
                            final_value = game_outcome_value if current_turn == chess.WHITE else -game_outcome_value

                # Final fallback if SF wasn't used or failed
                if final_value is None:
                    final_value = game_outcome_value if current_turn == chess.WHITE else -game_outcome_value

                # --- Encode Data ---
                encoded_position = encoder.encode_board(board) # Get (18, 8, 8) uint8 array
                encoded_move = encoder.encode_move(move)       # Get int

                # Append to lists
                positions_list.append(encoded_position)
                moves_list.append(encoded_move)
                values_list.append(final_value) # Append the final float value
                game_yielded_samples = True

                # Make move on board
                try:
                    board.push(move)
                    node = next_node
                except Exception:
                    break # Stop processing game if move fails

            if game_yielded_samples:
                games_processed_in_batch += 1 # Increment if moves were processed

    except Exception as worker_err:
        print(f"Critical Worker Error: {worker_err}")
        traceback.print_exc()
        # Return empty arrays on critical error to avoid crashing the main loop
        return np.array([], dtype=POS_DTYPE), np.array([], dtype=MOVE_DTYPE), np.array([], dtype=VALUE_DTYPE), 0
    finally:
        # Ensure engine is closed
        if engine:
            engine.quit()

    # Convert lists to NumPy arrays ---
    positions_batch_np = np.array(positions_list, dtype=POS_DTYPE)
    moves_batch_np = np.array(moves_list, dtype=MOVE_DTYPE)
    # Ensure values are float32 and have the correct shape (N, 1)
    values_batch_np = np.array(values_list, dtype=VALUE_DTYPE).reshape(-1, 1)

    return positions_batch_np, moves_batch_np, values_batch_np, games_processed_in_batch


# --- Main Processing Function ---
def run_combined_preencoding_hdf5(pgn_path, output_hdf5_path,
                                start_game_idx=1, end_game_idx=None,
                                num_workers=1, elo_floor=0, use_stockfish_eval=False,
                                games_per_batch=100,
                                stockfish_path=None, sf_eval_depth=10, sf_threads_per_worker=1):
    """
    Reads PGN, processes games (with optional live Stockfish eval),
    and writes directly to a chunked/compressed HDF5 file.
    """
    total_games_read_count = 0
    total_games_submitted_count = 0
    total_games_processed_count = 0
    total_positions_saved_count = 0
    encoder = ChessEncoder() # Instantiate encoder
    start_game_0idx = start_game_idx - 1
    end_game_0idx = end_game_idx - 1 if end_game_idx is not None else float('inf')

    output_dir = os.path.dirname(output_hdf5_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    print(f"Starting PGN processing and writing to HDF5")
    print(f"Input PGN: {pgn_path}")
    print(f"Output HDF5: {output_hdf5_path}")
    print(f"Processing game range (1-based): {start_game_idx} to {'End' if end_game_idx is None else end_game_idx}")
    print(f"ELO Floor: {elo_floor}, Use Stockfish Eval: {use_stockfish_eval}")
    if use_stockfish_eval:
        if stockfish_path and os.path.exists(stockfish_path):
             print(f"Live Stockfish Eval: ENABLED (Path: {stockfish_path}, Depth: {sf_eval_depth}, Threads/Worker: {sf_threads_per_worker})")
        else:
             print(f"Live Stockfish Eval: DISABLED (Stockfish path missing or invalid: '{stockfish_path}')")
             stockfish_path = None # Ensure path is None if invalid/missing
    print(f"Workers: {num_workers}, Games per worker batch: {games_per_batch}")
    print(f"HDF5 Chunks: Pos={POS_CHUNK}, Move={MOVE_CHUNK}, Val={VALUE_CHUNK}")

    start_time = time.time()

    try:
        # Open PGN and HDF5 files
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file, \
             h5py.File(output_hdf5_path, 'w') as f_out, \
             concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:

            # Dataset init
            print("Initializing HDF5 datasets...")
            pos_dset = f_out.create_dataset('positions', shape=(0, 18, 8, 8), maxshape=(None, 18, 8, 8),
                                          dtype=POS_DTYPE, chunks=POS_CHUNK, **COMPRESSION_OPTS)
            mov_dset = f_out.create_dataset('moves', shape=(0,), maxshape=(None,),
                                          dtype=MOVE_DTYPE, chunks=MOVE_CHUNK, **COMPRESSION_OPTS)
            val_dset = f_out.create_dataset('values', shape=(0, 1), maxshape=(None, 1),
                                          dtype=VALUE_DTYPE, chunks=VALUE_CHUNK, **COMPRESSION_OPTS)
            # Adding creation timestamp and source PGN
            f_out.attrs['creation_timestamp'] = time.time()
            f_out.attrs['source_pgn_file'] = os.path.basename(pgn_path)

            batch_futures = set()
            current_game_batch = []
            total_games_in_range = None # Progress bar total
            if end_game_idx is not None:
                  total_games_in_range = end_game_idx - start_game_idx + 1

            pbar_read = tqdm(desc="Scanning Games", unit="game", smoothing=0.1)
            # Process bar tracks games *processed* by workers, Write bar tracks *positions* written
            pbar_process = tqdm(total=total_games_in_range, desc="Processing Games", unit="game", disable=(total_games_in_range is None), smoothing=0.1)
            pbar_write = tqdm(desc="Writing Positions", unit="pos", smoothing=0.1)

            while True:
                # --- Process Completed Futures ---
                done_futures = {f for f in batch_futures if f.done()}
                if done_futures:
                    for future in done_futures:
                        try:
                            # Get NumPy arrays from worker
                            pos_batch_np, mov_batch_np, val_batch_np, games_proc_in_batch = future.result()
                            total_games_processed_count += games_proc_in_batch
                            pbar_process.update(games_proc_in_batch)

                            num_new_samples = len(mov_batch_np)
                            if num_new_samples > 0:
                                # Append to HDF5 sequentially
                                current_size = pos_dset.shape[0]
                                new_total_size = current_size + num_new_samples

                                pos_dset.resize(new_total_size, axis=0)
                                mov_dset.resize(new_total_size, axis=0)
                                val_dset.resize(new_total_size, axis=0)

                                pos_dset[current_size:] = pos_batch_np
                                mov_dset[current_size:] = mov_batch_np
                                val_dset[current_size:] = val_batch_np

                                # Maybe flush?
                                # if total_positions_saved_count % (SAMPLES_PER_TASK * num_workers) < num_new_samples: # Approx flush every N batches
                                #    f_out.flush()

                                total_positions_saved_count += num_new_samples
                                pbar_write.update(num_new_samples)
                                # --- End HDF5 Append ---

                            # Clean up returned data
                            del pos_batch_np, mov_batch_np, val_batch_np

                        except Exception as e:
                            print(f"\nError processing future result or writing to HDF5: {e}")
                            traceback.print_exc()
                            # Attempt to cancel remaining?
                            # for f_cancel in batch_futures: f_cancel.cancel()
                            # raise
                    batch_futures -= done_futures
                    gc.collect() # Manual garbage collection after processing batch results

                # Check if games are done reading
                if total_games_read_count > end_game_0idx and not batch_futures:
                    break

                # Read next game
                if total_games_read_count <= end_game_0idx:
                    game_start_offset = pgn_file.tell()
                    try:
                        game = chess.pgn.read_game(pgn_file)
                    except (ValueError, RuntimeError) as e:
                        print(f"\nWarning: Skipping invalid game at offset ~{game_start_offset}. Error: {e}")
                        # Recovery
                        try:
                             line = pgn_file.readline()
                             while line and not line.strip().startswith('[Event "'):
                                 if not line: break
                                 line = pgn_file.readline()
                             continue
                        except Exception as seek_e:
                             print(f"\nError trying to recover PGN stream: {seek_e}. Stopping.")
                             break
                        continue # Skip to next iteration

                    if game is None: # EOF
                        if not batch_futures: break # All read and all processed
                        else: time.sleep(0.05); continue # Wait for pending futures

                    # Add game to batch if in range
                    if total_games_read_count >= start_game_0idx:
                        current_game_batch.append(game)
                        total_games_submitted_count += 1
                    # Else: Game is before start index, skipped implicitly

                    total_games_read_count += 1
                    pbar_read.update(1)

                else: # Have read past end_game_idx, just wait for futures
                    if not batch_futures: break
                    time.sleep(0.05)
                    continue

                # --- Submit Batch ---
                should_submit = len(current_game_batch) >= games_per_batch # Indicates batch is ready
                is_last_batch = (game is None or total_games_read_count > end_game_0idx) and len(current_game_batch) > 0
                # Throttle submission slightly more if HDF5 writing is slow
                can_submit = len(batch_futures) < num_workers * 2 # Adjust multiplier if needed

                if (should_submit or is_last_batch) and can_submit:
                    if current_game_batch:
                        worker_args = (current_game_batch, elo_floor, use_stockfish_eval, encoder,
                                     stockfish_path, sf_eval_depth, sf_threads_per_worker)
                        future = executor.submit(_process_game_batch_hdf5_ready, worker_args)
                        batch_futures.add(future)
                        current_game_batch = [] # Reset batch
                elif (should_submit or is_last_batch) and not can_submit:
                    time.sleep(0.01) # If throttled, tiny pause

                # Break if EOF/End Index reached and the last batch was just submitted & processed
                if is_last_batch and not current_game_batch:
                    if not batch_futures: break # All batches processed
                    else: time.sleep(0.05) # Pause for last futures


            # --- Cleanup ---
            pbar_read.close()
            pbar_process.close()
            pbar_write.close()
            print("\nFinished reading PGN range and processing all batches.")
            print("Finalizing HDF5 file...")
            f_out.flush()

    except FileNotFoundError:
        print(f"Error: Input PGN file not found at {pgn_path}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        traceback.print_exc()
    finally:
        # --- Summary output after processing ---
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n--- Combined Pre-encoding Summary ---")
        print(f"Scanned {total_games_read_count} games from PGN.")
        print(f"Submitted {total_games_submitted_count} games (from index {start_game_idx} to {min(total_games_read_count, end_game_idx if end_game_idx else float('inf'))}) for processing.")
        print(f"Successfully processed {total_games_processed_count} valid games within range.")
        print(f"Saved {total_positions_saved_count} positions to {output_hdf5_path}")
        print(f"Total time: {duration:.2f} seconds")
        if total_positions_saved_count > 0 and duration > 0:
             positions_per_sec = total_positions_saved_count / duration
             print(f"Avg positions/sec: {positions_per_sec:.2f}")
        if os.path.exists(output_hdf5_path):
             final_size_gb = os.path.getsize(output_hdf5_path) / (1024**3)
             print(f"Final HDF5 file size: {final_size_gb:.3f} GB")


if __name__ == "__main__":
    mp.freeze_support() # multiprocessing on Windows
    print("Running Combined PGN-to-HDF5 Pre-encoder directly...")
    load_dotenv()

    # OS Environment path configuration
    pgn_file_path = os.getenv("PGN_PATH2")
    stockfish_path_env = os.getenv("STOCKFISH_PATH")
    games_dir = os.getenv("GAMES_PATH")

    if not pgn_file_path: sys.exit("Error: PGN_PATH environment variable not set.")
    if not games_dir: print("Warning: GAMES_PATH not set. Using current directory."); output_directory = "."
    else: output_directory = games_dir; os.makedirs(output_directory, exist_ok=True)
    if not stockfish_path_env: print("Warning: STOCKFISH_PATH not set. Live eval will be disabled if USE_SF_EVAL=True.")

    # Game Range to process
    START_GAME_INDEX = 1
    END_GAME_INDEX = 500_000

    # Output Filename
    range_str = f"{START_GAME_INDEX}-{END_GAME_INDEX}" if END_GAME_INDEX else f"{START_GAME_INDEX}-end"
    output_hdf5_filename = f"Games_{range_str}.hdf5"
    output_hdf5_full_path = os.path.join(output_directory, output_hdf5_filename)

    # Processing parameters
    NUM_WORKERS = max(1, mp.cpu_count() // 2 - 1) # Workers are lower for faster live SF eval
    GAMES_PER_BATCH = 50                     # Smaller batches -> more frequent updates to HDF5
    ELO_FLOOR = 1000
    USE_SF_EVAL = True

    # Stockfish Live Eval Config
    SF_EVAL_DEPTH = 6 # Depth low for speedier eval
    SF_THREADS_PER_WORKER = 1
    
    DIVISOR = 600 # Adjust for scaling CP scores to [-1, 1] range

    if not os.path.exists(pgn_file_path): sys.exit(f"Error: Input PGN file not found: '{pgn_file_path}'")
    if USE_SF_EVAL and not stockfish_path_env: print("Warning: USE_SF_EVAL is True, but STOCKFISH_PATH is not set.")

    # Calling main method to process PGN and write to the HDF5
    run_combined_preencoding_hdf5(
        pgn_path=pgn_file_path,
        output_hdf5_path=output_hdf5_full_path,
        start_game_idx=START_GAME_INDEX,
        end_game_idx=END_GAME_INDEX,
        num_workers=NUM_WORKERS,
        elo_floor=ELO_FLOOR,
        use_stockfish_eval=USE_SF_EVAL,
        games_per_batch=GAMES_PER_BATCH,
        stockfish_path=stockfish_path_env,
        sf_eval_depth=SF_EVAL_DEPTH,
        sf_threads_per_worker=SF_THREADS_PER_WORKER
    )

    print("HDF5 writing finished.")