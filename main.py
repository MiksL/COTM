import os
import torch
import traceback
from training.training import train_model
from playGame import play_human_vs_engine, play_engine_vs_engine, play_raw_vs_mcts, play_mcts_vs_external_engine, play_mcts_vs_stockfish_elo
from dotenv import load_dotenv
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_int_input, get_float_input, select_file, select_and_load_model, get_training_params

if __name__ == "__main__":
    load_dotenv()
    
    model = None
    
    while True:
        print("1. Train model from PGN")
        print("2. Train model from self-play")
        print("3. Play using a pre-trained model")
        userInput = input("Enter choice (or q to quit): ")

        if userInput.lower() == 'q': break
        if not userInput.isdigit(): print("Invalid input."); continue
        choice = int(userInput)

        # Train from existing games
        if choice == 1:
            print("\n--- Training from HDF5 ---")
            GAMES_PATH = os.getenv("GAMES_PATH")
            if not GAMES_PATH or not os.path.isdir(GAMES_PATH):
                print(f"Error: GAMES_PATH '{GAMES_PATH}' not found or not set in .env")
                continue
            
            hdf5_file_path = select_file(GAMES_PATH, ".hdf5", "HDF5")
            if not hdf5_file_path:
                continue
            
            trainingStart = time.time()
            model = train_model(
                data_path=hdf5_file_path,
                epochs=10,
                chunks_per_batch=4,
                learning_rate=0.5e-3,
                val_split=0.1,
                patience=5,
            )
            
            gc.collect()
            trainingEnd = time.time()
            print(f"Training time: {(trainingEnd - trainingStart) / 60:.2f} minutes")
            
            torch.save(model.state_dict(), "TestBotFallbackEG.pth")
            
            saveModel = input("Save the model? (y/n)")
            if saveModel.lower() == 'y':
                modelname = input("Enter model name: ")
                try:
                    torch.save(model.state_dict(), "models/" + modelname + ".pth")
                except:
                    torch.save(model.state_dict(), "models/" + modelname + ".pth")

        # Train from self-play
        elif choice == 2:
            print("\n--- Training from Self-Play ---")
            
            GAMES_PATH = os.getenv("SELF_PLAY_GAMES_PATH")
            
            # Fix for WSL paths
            if GAMES_PATH and GAMES_PATH.startswith('mnt/') and not GAMES_PATH.startswith('/'):
                GAMES_PATH = '/' + GAMES_PATH
                os.environ['SELF_PLAY_GAMES_PATH'] = GAMES_PATH
                print(f"Converted to absolute path: {GAMES_PATH}")
                
            if not GAMES_PATH or not os.path.isdir(GAMES_PATH):
                print(f"Error: SELF_PLAY_GAMES_PATH '{GAMES_PATH}' not found or not set in .env")
                continue
                
            mode = input("Choose training mode:\n1. Continue training from existing model and data\n2. Generate new self-play games\nChoice: ")
            
            if mode == "1":
                # Select existing model (optional)
                model_path = select_file("models", ".pth", "model")
                
                # Select existing HDF5 file
                hdf5_path = select_file(GAMES_PATH, ".hdf5", "HDF5")
                if not hdf5_path:
                    print("No HDF5 file selected. Please select a file to train from.")
                    continue
                
                # Get training parameters
                print("\nEnter training parameters:")
                params = get_training_params()
                
                trainingStart = time.time()
                
                model = train_model(
                    data_path=hdf5_path,
                    epochs=params['epochs'],
                    chunks_per_batch=params['chunks_per_batch'],
                    learning_rate=params['learning_rate'],
                    val_split=params['val_split'],
                    initial_model_path=model_path,
                    patience=params['patience']
                )
                
                gc.collect()
                trainingEnd = time.time()
                print(f"Training time: {(trainingEnd - trainingStart) / 60:.2f} minutes")
                
                timestamp = int(time.time())
                fallback_path = f"models/SelfPlayTrained_{timestamp}.pth"
                torch.save(model.state_dict(), fallback_path)
                print(f"Model saved to {fallback_path} (fallback)")
                
                saveModel = input("Save the model with a custom name? (y/n): ")
                if saveModel.lower() == 'y':
                    modelname = input("Enter model name: ")
                    try:
                        save_path = os.path.join("models", f"{modelname}.pth")
                        torch.save(model.state_dict(), save_path)
                        print(f"Model saved to {save_path}")
                    except Exception as e:
                        print(f"Error saving model: {e}")
                        print(f"Model is still available at {fallback_path}")
            
            elif mode == "2":
                print("This option would generate new self-play games.")
                print("To generate self-play games, run gameGenerationTest.py directly.")
                print("It will create HDF5 files in your SELF_PLAY_GAMES_PATH directory.")
                print("You can then use option 1 to train on those generated games.")
            
            else:
                print("Invalid mode selected, returning to main menu.")

        # Load Model & Play
        elif choice == 3:
            print("\n--- Load Model & Play ---")
            print("1. Play against engine (Human vs MCTS Engine)")
            print("2. Watch engines play (MCTS Engine vs MCTS Engine)")
            print("3. Compare Raw NN vs MCTS Engine (using same model)") 
            print("4. Play MCTS Engine vs External UCI Engine")
            print("5. Play MCTS Engine vs Stockfish (ELO Limited)")
            models_choice = input("Enter choice: ")

            if not models_choice.isdigit(): print("Invalid choice."); continue
            play_mode = int(models_choice)
            if play_mode not in [1, 2, 3, 4, 5]: print("Invalid choice."); continue

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\nUsing device: {device}")

            # Get MCTS Simulation Count
            num_simulations = get_int_input("Enter MCTS simulations per move (default 400): ", 400, 1)
            
            use_sf = input("Use Stockfish for evaluation comparison? (y/n): ").lower() == 'y'

            # Mode 1: Human vs Engine
            if play_mode == 1:
                model_state_dict, model_name = select_and_load_model(device)
                if model_state_dict is None:
                    continue
                play_human_vs_engine(model_state_dict, num_simulations, use_stockfish=use_sf)

            # Mode 2: Engine vs Engine
            elif play_mode == 2:
                print("Select model 1 (White):")
                state_dict1, model1_name = select_and_load_model(device)
                if state_dict1 is None:
                    continue
                
                print("Select model 2 (Black):")
                state_dict2, model2_name = select_and_load_model(device)
                if state_dict2 is None:
                    continue

                pause = input("Pause after each move? (y/n): ").lower() == 'y'
                play_engine_vs_engine(state_dict1, state_dict2, num_simulations,
                                      use_stockfish=use_sf, pause_after_move=pause,
                                      model1_name=model1_name, model2_name=model2_name)

            # Mode 3: Raw NN vs MCTS
            elif play_mode == 3:
                model_state_dict, model_name = select_and_load_model(device)
                if model_state_dict is None:
                    continue

                raw_plays_white = input("Should Raw NN play as White? (y/n): ").lower() == 'y'
                pause = input("Pause after each move? (y/n): ").lower() == 'y'

                play_raw_vs_mcts(model_state_dict, num_simulations,
                                 raw_plays_white=raw_plays_white,
                                 use_stockfish=use_sf,
                                 pause_after_move=pause,
                                 model_name=model_name)

            # Mode 4: MCTS Engine vs External UCI Engine
            elif play_mode == 4:
                model_state_dict, model_name = select_and_load_model(device)
                if model_state_dict is None:
                    continue

                uci_engine_path = select_file("engines", "", "UCI engine")
                if not uci_engine_path or not os.path.isfile(uci_engine_path):
                    print("Error: UCI engine not found.")
                    continue

                pause = input("Pause after each move? (y/n): ").lower() == 'y'

                game_params = {
                    "model_state_dict": model_state_dict,
                    "exploration_weight": 1.4,
                    "uci_engine_path": uci_engine_path,
                    "use_stockfish": use_sf,
                    "pause_after_move": pause,
                    "model_name": model_name,
                }
                
                # Play 4 games: 2 MCTS White, 2 MCTS Black
                for i, mcts_white in enumerate([True, True, False, False], 1):
                    color = "White" if mcts_white else "Black"
                    print(f"\n--- Game {i} of 4 (MCTS as {color}) ---")
                    play_mcts_vs_external_engine(**game_params, num_simulations=num_simulations, mcts_plays_white=mcts_white)

            # Mode 5: MCTS Engine vs Stockfish (ELO Limited)
            elif play_mode == 5:
                if not (os.getenv("STOCKFISH_PATH") and os.path.exists(os.getenv("STOCKFISH_PATH"))):
                    print("Error: STOCKFISH_PATH not set or invalid in .env file.")
                    continue
                
                # Model selection
                model_state_dict, model_name = select_and_load_model(device)
                if model_state_dict is None:
                    continue

                # Stockfish configuration
                stockfish_elo = get_int_input("Stockfish ELO limit (e.g., 1500): ", min_val=1)
                sf_think_time = get_float_input("Stockfish think time in seconds (default 0.1): ", 0.1)
                
                # Game configuration
                total_game_sets = get_int_input("Total number of game sets to play: ", min_val=1)
                
                # Simulation configuration
                vary_sims = input("Vary MCTS simulations across sets? (y/n): ").lower() == 'y'
                if vary_sims:
                    start_sims = get_int_input(f"Starting simulations (default {num_simulations}): ", num_simulations, 1)
                    end_sims = get_int_input(f"Ending simulations (default {start_sims + 400}): ", start_sims + 400, 1)
                    step_size = get_int_input("Step size (default 200): ", 200, 1)
                    sim_counts = list(range(start_sims, end_sims + 1, step_size))
                    
                    # Distribute game sets evenly across simulation counts
                    sets_per_sim = total_game_sets // len(sim_counts)
                    remaining_sets = total_game_sets % len(sim_counts)
                    
                    game_configs = []
                    for i, sim_count in enumerate(sim_counts):
                        sets_for_this_sim = sets_per_sim + (1 if i < remaining_sets else 0)
                        for _ in range(sets_for_this_sim):
                            game_configs.append(sim_count)
                else:
                    game_configs = [num_simulations] * total_game_sets
                    
                    print(f"Total game sets: {len(game_configs)}")
                    if vary_sims:
                        sim_distribution = {sim: game_configs.count(sim) for sim in set(game_configs)}
                        print(f"Simulation distribution: {sim_distribution}")
                    
                    # Execute all games with 4 threads (simplified)
                    all_results = []
                    
                    for set_num, sim_count in enumerate(game_configs):
                        print(f"\n{'='*50}")
                        print(f"Running set {set_num + 1}/{len(game_configs)} with {sim_count} simulations")
                        print(f"{'='*50}")
                        
                        base_params = {
                            "model_state_dict": model_state_dict,
                            "exploration_weight": 1.4,
                            "stockfish_elo_limit": stockfish_elo,
                            "use_aux_stockfish_eval": use_sf,
                            "model_name": model_name,
                            "elo_limited_stockfish_think_time": sf_think_time,
                            "num_simulations": sim_count
                    }
                    
                    # Prepare 4 games (2 white, 2 black)
                    all_games = []
                    for mcts_white in [True, True, False, False]:
                        all_games.append({**base_params, "mcts_plays_white": mcts_white})
                    
                    # Run 4 games in parallel
                    set_results = []
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [executor.submit(play_mcts_vs_stockfish_elo, **params) for params in all_games]
                        
                        for i, future in enumerate(as_completed(futures)):
                            try:
                                result, mcts_white = future.result()
                                set_results.append({"result": result, "mcts_white": mcts_white})
                                print(f"Game {i+1}/4 completed: {result}")
                            except Exception as e:
                                print(f"Game {i+1}/4 failed: {e}")
                                set_results.append({"result": "ERROR", "mcts_white": None})
                        
                        # Calculate scores for this set
                        scores = {"mcts_wins": 0, "sf_wins": 0, "draws": 0, 
                                 "mcts_white_wins": 0, "mcts_black_wins": 0}
                        
                        for game_result in set_results:
                            result = game_result["result"]
                            mcts_white = game_result["mcts_white"]
                            
                            if result == "1-0":  # White won
                                if mcts_white:
                                    scores["mcts_wins"] += 1
                                    scores["mcts_white_wins"] += 1
                                else:
                                    scores["sf_wins"] += 1
                            elif result == "0-1":  # Black won
                                if not mcts_white:
                                    scores["mcts_wins"] += 1
                                    scores["mcts_black_wins"] += 1
                                else:
                                    scores["sf_wins"] += 1
                            elif result == "1/2-1/2":
                                scores["draws"] += 1
                        
                        all_results.append({"sims": sim_count, "scores": scores})
                        
                        # Print set summary
                        total_games = scores["mcts_wins"] + scores["sf_wins"] + scores["draws"]
                        if total_games > 0:
                            mcts_score = scores["mcts_wins"] + 0.5 * scores["draws"]
                            win_rate = mcts_score / total_games
                            print(f"Set {set_num + 1} ({sim_count} sims): "
                                  f"MCTS {mcts_score}/{total_games} ({win_rate:.1%})")
                    
                    # Final summary
                    print(f"\n{'='*60}")
                    print("FINAL MATCH SUMMARY")
                    print(f"{'='*60}")
                    
                    # Aggregate by simulation count
                    sim_totals = {}
                    for result in all_results:
                        sim_count = result["sims"]
                        if sim_count not in sim_totals:
                            sim_totals[sim_count] = {"mcts_wins": 0, "sf_wins": 0, "draws": 0, 
                                                   "mcts_white_wins": 0, "mcts_black_wins": 0}
                        
                        scores = result["scores"]
                        for key in sim_totals[sim_count]:
                            sim_totals[sim_count][key] += scores[key]
                    
                    for sim_count in sorted(sim_totals.keys()):
                        scores = sim_totals[sim_count]
                        total_games = scores["mcts_wins"] + scores["sf_wins"] + scores["draws"]
                        if total_games > 0:
                            mcts_score = scores["mcts_wins"] + 0.5 * scores["draws"]
                            win_rate = mcts_score / total_games
                            print(f"\nMCTS {sim_count} sims vs Stockfish ELO {stockfish_elo}:")
                            print(f"  MCTS: {scores['mcts_wins']} wins ({scores['mcts_white_wins']}W, {scores['mcts_black_wins']}B)")
                            print(f"  Stockfish: {scores['sf_wins']} wins")
                            print(f"  Draws: {scores['draws']}")
                            print(f"  MCTS Score: {mcts_score}/{total_games} ({win_rate:.1%})")

        else:
            print("Invalid input. Please enter a valid choice number or 'q'.")

    print("\nExiting program.")