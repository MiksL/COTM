import os
import torch
import traceback
from training import train_model
from playGame import play_human_vs_engine, play_engine_vs_engine, play_raw_vs_mcts, RawNNEngine, MCTSEngine
from selfPlay import SelfPlayChess

from readGames import PGNReader
from dotenv import load_dotenv
import gc
import time

if __name__ == "__main__":
    load_dotenv()
    ''' 
    1. Read existing games to be used for training
    2. Turn games into usable data - board state encoding 
    '''
    
    # TODO - look at ways to improve game encoding
    # TODO - pre-encode and save games to remove redundant processing
    
    # Define the model var
    model = None
    
    # userInput loop
    while True:
        print("1. Train model from PGN")
        print("2. Train model from self-play (Not Implemented)")
        print("3. Play using a pre-trained model")
        userInput = input("Enter choice (or q to quit): ")

        if userInput.lower() == 'q': break
        if not userInput.isdigit(): print("Invalid input."); continue
        choice = int(userInput)

        # --- Train ---
        if choice == 1:
            # Train from existing games
            print("\n--- Training from HDF5 ---")
            GAMES_PATH = os.getenv("GAMES_PATH")
            if not GAMES_PATH or not os.path.isdir(GAMES_PATH):
                print(f"Error: GAMES_PATH '{GAMES_PATH}' not found or not set in .env")
                continue
            
            available_hdf5 = [f for f in os.listdir(GAMES_PATH) if f.endswith(".hdf5")]
            if not available_hdf5:
                print(f"No HDF5 files found in {GAMES_PATH}")
                continue

            print("Available HDF5 files:")
            for i, fname in enumerate(available_hdf5):
                print(f"{i+1}. {fname}")
            file_choice = int(input("Select HDF5 file to train from: ")) - 1
            
            hdf5_file_path = os.path.join(GAMES_PATH, available_hdf5[file_choice])
            
            ''' 
            3. Train the models
            3.1. Learns from the games generated/given
            3.2. Learns from self-play
            '''
            
            trainingStart = time.time()
            # 3.1
            # Train the model
            model = train_model(
                data_path=hdf5_file_path,
                epochs=25,
                chunks_per_batch=2,
                learning_rate=0.5e-3,
                val_split=0.1,
            )
            
            #del positions, moves, values
            gc.collect()
            
            # End of training
            trainingEnd = time.time()
            
            # Training time in minutes
            print(f"Training time: {(trainingEnd - trainingStart) / 60:.2f} minutes")
            
            # Save model - fallback case
            torch.save(model.state_dict(), "TestBotFallbackEG.pth")
            
            saveModel = input("Save the model? (y/n)")
            
            if saveModel.lower() == 'y':
                modelname = input("Enter model name: ")
                # Try to sdave the model in the models directory, else save in the working directory
                try:
                    torch.save(model.state_dict(), "models/" + modelname + ".pth")
                except: # If error - save in the working directory
                    torch.save(model.state_dict(), "models/" + modelname + ".pth")
                pass
            else:
                pass
        # --- Self Play ---
        elif choice == 2:
            # Train from self-play
                print("Training from self-play")

                from neuralNetwork import ChessNN
                from encoding import ChessEncoder
                from selfPlay import SelfPlayChess
                from training import train_sp
                
                # Example usage
                model = ChessNN()
                encoder = ChessEncoder() # TODO: Fix usage of new encoder - precomputed map not found

                # Run self-play to generate training data
                self_play = SelfPlayChess(model, encoder, 100)
                dataset = self_play.run_self_play()
                

                # Train the model using the generated dataset
                train_sp(model, dataset, epochs=10, batch_size=1024, learning_rate=1e-3)
                
                # Prompt user to save the model
                saveModel = input("Save the model? (y/n)")
                if saveModel.lower() == 'y':
                    # Save the model in the models directory
                    torch.save(model.state_dict(), "models/TestBotSP.pth")
                    pass
                else:
                    pass

        # --- Load Model & Play ---
        elif choice == 3:
            print("\n--- Load Model & Play ---")
            print("1. Play against engine (Human vs MCTS Engine)")
            print("2. Watch engines play (MCTS Engine vs MCTS Engine)")
            print("3. Compare Raw NN vs MCTS Engine (using same model)") 
            models_choice = input("Enter choice: ")

            if not models_choice.isdigit(): print("Invalid choice."); continue
            play_mode = int(models_choice)
            if play_mode not in [1, 2, 3]: print("Invalid choice."); continue 

            # --- List Models ---
            try:
                model_dir = "models"
                if not os.path.isdir(model_dir): print(f"Directory '{model_dir}' not found."); input("Press Enter..."); continue
                available_models = sorted([m for m in os.listdir(model_dir) if m.endswith(".pth")])
                if not available_models: print(f"Error: No models (.pth files) found in '{model_dir}'."); input("Press Enter..."); continue
                print("\nAvailable models:")
                for i, model_name in enumerate(available_models): print(f"{i+1}. {model_name}")
            except Exception as e: print(f"Error listing models: {e}"); input("Press Enter..."); continue

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"\nUsing device: {device}")

            # --- Get MCTS Simulation Count ---
            # Moved outside specific modes as it's needed for modes 1, 2, and 3 (for the MCTS side)
            try:
                sims_str = input(f"Enter MCTS simulations per move for MCTS engine(s) (e.g., 100-1600+): ")
                num_simulations = int(sims_str)
                if num_simulations <= 0: raise ValueError("Simulations must be positive.")
            except ValueError: print("Invalid number, using default 400 simulations."); num_simulations = 400

            # --- Ask about Stockfish ---
            use_sf = input("Use Stockfish for evaluation comparison? (y/n): ").lower() == 'y'

            # --- Mode 1: Human vs Engine ---
            if play_mode == 1:
                try:
                    modelIndex = int(input("Select model to play against (enter number): ")) - 1
                    if not 0 <= modelIndex < len(available_models): print("Invalid model number."); continue
                    selected_model_name = available_models[modelIndex]
                    model_path = os.path.join(model_dir, selected_model_name)
                    print(f"Loading model: {selected_model_name}...")
                    model_state_dict = torch.load(model_path, map_location=device)
                    print("Model state loaded.")
                    play_human_vs_engine(model_state_dict, num_simulations, use_stockfish=use_sf)
                # ... (Error handling) ...
                except (ValueError, IndexError): print("Invalid input.")
                except FileNotFoundError: print(f"Error: Model file not found '{model_path}'.")
                except Exception as e: print(f"Unexpected error setting up game: {e}"); traceback.print_exc(); input("\nPress Enter...")

            # --- Mode 2: Engine vs Engine ---
            elif play_mode == 2:
                try:
                    modelIndex1 = int(input("Select model 1 (White, enter number): ")) - 1
                    modelIndex2 = int(input("Select model 2 (Black, enter number): ")) - 1
                    if not (0 <= modelIndex1 < len(available_models) and 0 <= modelIndex2 < len(available_models)): print("Invalid model number."); continue

                    model1_name = available_models[modelIndex1]
                    model2_name = available_models[modelIndex2]
                    model1_path = os.path.join(model_dir, model1_name)
                    model2_path = os.path.join(model_dir, model2_name)
                    print(f"Model 1 (W): {model1_name} | Model 2 (B): {model2_name}")
                    if modelIndex1 == modelIndex2: print("Note: Selected models are the same.")

                    print(f"Loading model 1 state_dict..."); state_dict1 = torch.load(model1_path, map_location=device)
                    print(f"Loading model 2 state_dict..."); state_dict2 = torch.load(model2_path, map_location=device)
                    print("Model states loaded.")

                    pause = input("Pause after each move? (y/n): ").lower() == 'y'
                    play_engine_vs_engine(state_dict1, state_dict2, num_simulations,
                                          use_stockfish=use_sf, pause_after_move=pause,
                                          model1_name=model1_name, model2_name=model2_name)
                # ... (Error handling) ...
                except (ValueError, IndexError): print("Invalid input.")
                except FileNotFoundError: print(f"Error: Model file not found.")
                except Exception as e: print(f"Unexpected error setting up game: {e}"); traceback.print_exc(); input("\nPress Enter...")

            # --- Mode 3: Raw NN vs MCTS ---
            elif play_mode == 3:
                try:
                    modelIndex = int(input("Select ONE model for both players (enter number): ")) - 1
                    if not 0 <= modelIndex < len(available_models): print("Invalid model number."); continue

                    selected_model_name = available_models[modelIndex]
                    model_path = os.path.join(model_dir, selected_model_name)
                    print(f"Loading model: {selected_model_name}...")
                    model_state_dict = torch.load(model_path, map_location=device) # Load only one state dict
                    print("Model state loaded.")

                    # Ask which player uses Raw NN
                    raw_white_choice = input("Should Raw NN play as White? (y/n): ").lower()
                    raw_plays_white = (raw_white_choice == 'y')

                    pause = input("Pause after each move? (y/n): ").lower() == 'y'

                    # Call the NEW game function
                    play_raw_vs_mcts(model_state_dict, num_simulations,
                                     raw_plays_white=raw_plays_white,
                                     use_stockfish=use_sf,
                                     pause_after_move=pause,
                                     model_name=selected_model_name)

                except (ValueError, IndexError): print("Invalid input.")
                except FileNotFoundError: print(f"Error: Model file not found '{model_path}'.")
                except Exception as e: print(f"Unexpected error setting up game: {e}"); traceback.print_exc(); input("\nPress Enter...")

        else:
             print("Invalid input. Please enter a valid choice number or 'q'.")

    print("\nExiting program.")
