import os
import chess.pgn
from collections import defaultdict
import re

GAMES_DIRECTORY = "mcts_vs_stockfish_2750_elo"  # Directory with PGN files
MCTS_PLAYER_NAME_TAG = "MCTS"  # Identifier for MCTS player in PGN White/Black tags
SIMULATION_HEADER_TAGS = ["Simulations", "MCTSSimulations", "MCTS_Simulations"] # PGN tags for simulation count

def parse_simulations_from_header(headers):
    """Parse the number of simulations from game headers if available."""
    for tag in SIMULATION_HEADER_TAGS:
        if tag in headers:
            try:
                return int(headers[tag])
            except ValueError:
                # print(f"Warning: Could not parse simulation count from header '{tag}': {headers[tag]}")
                pass
    return None

def get_mcts_player_and_simulations(headers):
    """
    Determine which player is MCTS and the number of simulations from PGN headers.
    Returns: (mcts_is_white: bool|None, simulations: int|None)
    """
    white_player = headers.get("White", "").upper()
    black_player = headers.get("Black", "").upper()
    simulations = parse_simulations_from_header(headers)

    mcts_is_white = None
    if MCTS_PLAYER_NAME_TAG.upper() in white_player:
        mcts_is_white = True
    elif MCTS_PLAYER_NAME_TAG.upper() in black_player:
        mcts_is_white = False
    else:
        # Fallback: Try to extract from player name format "MCTS_SIMS_XXX"
        sim_match_white = re.search(r"MCTS_SIMS_(\d+)", white_player, re.IGNORECASE)
        sim_match_black = re.search(r"MCTS_SIMS_(\d+)", black_player, re.IGNORECASE)
        if sim_match_white:
            mcts_is_white = True
            if simulations is None: simulations = int(sim_match_white.group(1))
        elif sim_match_black:
            mcts_is_white = False
            if simulations is None: simulations = int(sim_match_black.group(1))

    if mcts_is_white is None:
        return None, simulations # MCTS player couldn't be identified

    if simulations is None:
        # Fallback: Try to extract from player name format "(Sims: XXX)"
        player_to_check_name = white_player if mcts_is_white else black_player
        match = re.search(r"\(SIMS:\s*(\d+)\)", player_to_check_name, re.IGNORECASE) # More robust regex
        if match:
            try:
                simulations = int(match.group(1)) # Group 1 is the digits
            except ValueError:
                pass # Error parsing this format
        
    # if simulations is None: # Optional: log if still not found
        # print(f"Warning: Could not determine simulation count for MCTS. White: '{headers.get('White', '')}', Black: '{headers.get('Black', '')}'")

    return mcts_is_white, simulations

def analyze_games():
    """Analyzes PGN files in the GAMES_DIRECTORY and aggregates results by simulation count."""
    if not os.path.isdir(GAMES_DIRECTORY):
        print(f"Error: Games directory '{GAMES_DIRECTORY}' not found.")
        return

    # results[sim_count][category] where category includes win/loss/draw counts
    results = defaultdict(lambda: defaultdict(int))
    files_processed = 0
    games_analyzed = 0
    games_skipped_mcts_id = 0
    games_skipped_sim_id = 0

    for filename in os.listdir(GAMES_DIRECTORY):
        if filename.endswith(".pgn"):
            filepath = os.path.join(GAMES_DIRECTORY, filename)
            files_processed += 1
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None: break # End of file or invalid PGN entry
                        
                        games_analyzed += 1
                        headers = game.headers
                        result_str = headers.get("Result", "*")

                        mcts_is_white, simulations = get_mcts_player_and_simulations(headers)

                        if mcts_is_white is None:
                            games_skipped_mcts_id += 1
                            continue
                        
                        if simulations is None:
                            games_skipped_sim_id += 1
                            continue
                        
                        current_sim_stats = results[simulations]
                        current_sim_stats["total_games_for_sim_count"] += 1

                        if result_str == "1-0": # White won
                            if mcts_is_white:
                                current_sim_stats["mcts_wins_as_white"] += 1
                            else:
                                current_sim_stats["opponent_wins_vs_mcts_black"] += 1
                        elif result_str == "0-1": # Black won
                            if not mcts_is_white:
                                current_sim_stats["mcts_wins_as_black"] += 1
                            else:
                                current_sim_stats["opponent_wins_vs_mcts_white"] += 1
                        elif result_str == "1/2-1/2": # Draw
                            if mcts_is_white:
                                current_sim_stats["mcts_draws_as_white"] += 1
                            else:
                                current_sim_stats["mcts_draws_as_black"] += 1
                        elif result_str == "*":
                            current_sim_stats["unfinished_games"] += 1
                        else:
                            current_sim_stats["unknown_results"] += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                # Attempt to attribute error to sim count if known, else a generic error bucket
                sim_count_for_error = locals().get('simulations', "unknown_sim_errors")
                results[sim_count_for_error]["processing_errors"] += 1

    print(f"\n--- Overall Analysis Summary ---")
    print(f"Processed {files_processed} PGN files, analyzed {games_analyzed} games.")
    if games_skipped_mcts_id > 0:
        print(f"Skipped {games_skipped_mcts_id} games: MCTS player not identified.")
    if games_skipped_sim_id > 0:
        print(f"Skipped {games_skipped_sim_id} games: Simulation count not identified for MCTS.")

    # Sort simulation counts for consistent output order (numbers first, then strings like "unknown_sim_errors")
    sorted_simulation_keys = sorted(results.keys(), key=lambda x: (isinstance(x, str), x))

    for sim_count_key in sorted_simulation_keys:
        stats = results[sim_count_key]
        if sim_count_key == "unknown_sim_errors":
            print(f"\n--- Errors (Simulation Count Unknown During Processing) ---")
            print(f"  File/Game Processing Errors: {stats.get('processing_errors', 0)}")
            continue

        print(f"\n--- Results for MCTS with {sim_count_key} Simulations ---")
        
        mcts_total_wins = stats["mcts_wins_as_white"] + stats["mcts_wins_as_black"]
        mcts_total_draws = stats["mcts_draws_as_white"] + stats["mcts_draws_as_black"]
        opponent_total_wins = stats["opponent_wins_vs_mcts_white"] + stats["opponent_wins_vs_mcts_black"]
        
        total_completed_games_for_sim = mcts_total_wins + opponent_total_wins + mcts_total_draws
        
        print(f"  Total Games Tracked for this Sim Count: {stats['total_games_for_sim_count']}")
        # if total_completed_games_for_sim < stats['total_games_for_sim_count']:
             # print(f"  (Note: Check for unfinished/unknown result games below if numbers don't match)")

        print(f"  MCTS Wins: {mcts_total_wins}")
        print(f"    As White: {stats['mcts_wins_as_white']}")
        print(f"    As Black: {stats['mcts_wins_as_black']}")
        
        print(f"  Opponent Wins: {opponent_total_wins}")
        print(f"    When MCTS was White: {stats['opponent_wins_vs_mcts_white']}")
        print(f"    When MCTS was Black: {stats['opponent_wins_vs_mcts_black']}")

        print(f"  Draws: {mcts_total_draws}")
        print(f"    When MCTS was White: {stats['mcts_draws_as_white']}")
        print(f"    When MCTS was Black: {stats['mcts_draws_as_black']}")

        if stats.get("processing_errors", 0) > 0:
            print(f"  File/Game Processing Errors for this sim count: {stats['processing_errors']}")
        if stats.get("unfinished_games", 0) > 0:
            print(f"  Unfinished Games (*): {stats['unfinished_games']}")
        if stats.get("unknown_results", 0) > 0:
            print(f"  Games with Unknown Results: {stats['unknown_results']}")

        if total_completed_games_for_sim > 0:
            mcts_score = mcts_total_wins + (0.5 * mcts_total_draws)
            win_percentage = mcts_score / total_completed_games_for_sim
            print(f"  MCTS Score (completed games): {mcts_score}/{total_completed_games_for_sim} ({win_percentage:.2%})")
        else:
            print(f"  MCTS Score: N/A (No completed games for this simulation count)")
            
    if not results:
        print("No PGN files found or no games could be appropriately analyzed from the PGNs.")

if __name__ == "__main__":
    # Check if the target directory exists, if not, inform the user.
    if not os.path.exists(GAMES_DIRECTORY) or not os.listdir(GAMES_DIRECTORY):
        print(f"INFO: The specified GAMES_DIRECTORY ('{GAMES_DIRECTORY}') is empty or does not exist.")
        print("Please ensure PGN files are present in this directory to run the analysis.")
        # Example: You might want to exit or provide instructions to create dummy files manually if needed for testing.
    else:
        analyze_games() 