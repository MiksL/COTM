import os
import torch
import traceback

from training import train_model
# TODO: Improve playGame.py - implement inference usage
from playGame import play_human_vs_engine, play_engine_vs_engine

from readGames import PGNReader
from dotenv import load_dotenv

import gc

import time

if __name__ == "__main__":
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
        # user input to train an existing game/self-play model or load a pre-trained model from the models directory
        userInput = input("1. Train from existing games\n2. Train from self-play\n3. Load a pre-trained model\n")
        if userInput.isdigit():
            if int(userInput) == 1:
                # Train from existing games
                PGN_PATH = os.getenv("PGN_PATH") # Get PGN path from environment variable - points to game file
                sampler = PGNReader(PGN_PATH)
                
                # TODO - Chunk files, place in GAMES_PATH
                positions_path, moves_path, values_path = sampler.preprocess_games(150000, 12, 2000)
                
                # Close the file, release resources - sampler not needed after game pre-processing
                del sampler
                gc.collect()
                
                ''' 
                3. Train the models
                3.1. Learns from the games generated/given
                3.2. Learns from self-play
                '''
                
                # TODO - use numpy.load for memmap file loading mmap_mode='r' for batch processing and reading into memory
                # TODO - filter out games with low ELO ratings
                
                trainingStart = time.time()
                # 3.1
                # Train the model
                model = train_model(
                    positions=positions_path,
                    moves=moves_path,
                    values=values_path,
                    epochs=10,
                    batch_size=4096,
                    learning_rate=1e-3,
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

            elif int(userInput) == 2:
                # Train from self-play
                print("Training from self-play")
                
                from neuralNetworkSP import SelfPlayChess, ChessNN, ChessEncoder, train
                
                # Example usage
                model = ChessNN()
                encoder = ChessEncoder()

                # Run self-play to generate training data
                self_play = SelfPlayChess(model, encoder, 100)
                dataset = self_play.run_self_play()

                # Train the model using the generated dataset
                train(model, dataset, epochs=10, batch_size=1024, learning_rate=1e-3)
                
                # Prompt user to save the model
                saveModel = input("Save the model? (y/n)")
                if saveModel.lower() == 'y':
                    # Save the model in the models directory
                    torch.save(model.state_dict(), "models/TestBotSP.pth")
                    pass
                else:
                    pass
                
                
            elif int(userInput) == 3:
                models_choice = input("Load 1 model (play against) or 2 models (engine vs engine)? (1/2): ")

                if models_choice.isdigit():
                    num_models_to_load = int(models_choice)

                    # List available models in the directory
                    try:
                        model_dir = "models"
                        if not os.path.isdir(model_dir):
                            print(f"Directory '{model_dir}' not found.")
                            input("Press Enter to continue...")
                            continue

                        available_models = [model for model in os.listdir(model_dir) if model.endswith(".pth")]
                        if not available_models:
                            print(f"Error: No models found in '{model_dir}'.")
                            input("Press Enter to continue...")
                            continue

                        print("\nAvailable models:")
                        for i, model_name in enumerate(available_models):
                            print(f"{i+1}. {model_name}")

                    except Exception as e:
                        print(f"Error listing models: {e}")
                        input("Press Enter to continue...")
                        continue

                    # Using CUDA for inference if available
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"\nUsing device: {device}")

                    # --- Load 1 Model ---
                    if num_models_to_load == 1:
                        try:
                            modelIndex = int(input("Model to play against: ")) - 1
                            if not 0 <= modelIndex < len(available_models):
                                print("Invalid model")
                                continue

                            selected_model_name = available_models[modelIndex]
                            model_path = os.path.join(model_dir, selected_model_name)

                            print(f"Loading model: {selected_model_name}...")
                            # Load state dictionary to device
                            model_state_dict = torch.load(model_path, map_location=device)

                            think_time_str = input("Enter engine think time per move in seconds (eg, 2.0/5.4): ")
                            try:
                                think_time = float(think_time_str)
                            except ValueError:
                                print("Invalid time, using 2.0s")
                                think_time = 2.0

                            # --- Call game method for playGame ---
                            play_human_vs_engine(model_state_dict, think_time)

                        # Error handling
                        except ValueError:
                            print("Invalid input.")
                        except IndexError:
                            print("Invalid model number selected.")
                        except FileNotFoundError:
                            print(f"Error: Model file not found '{model_path}'.")
                        except Exception as e:
                            print(f"Unexpected error: {e}")
                            traceback.print_exc()
                            input("\nPress Enter to continue...")


                    # --- Load 2 Models (Engine vs Engine) ---
                    elif num_models_to_load == 2:
                        try:
                            modelIndex1 = int(input("Select model 1 (White, enter number): ")) - 1
                            modelIndex2 = int(input("Select model 2 (Black, enter number): ")) - 1

                            if not (0 <= modelIndex1 < len(available_models) and 0 <= modelIndex2 < len(available_models)):
                                print("Invalid model number")
                                continue
                            if modelIndex1 == modelIndex2:
                                print("Selected models are the same")

                            model1_name = available_models[modelIndex1]
                            model2_name = available_models[modelIndex2]
                            model1_path = os.path.join(model_dir, model1_name)
                            model2_path = os.path.join(model_dir, model2_name)

                            # --- Load State Dicts ---
                            print(f"Loading model 1 (White): {model1_name}...")
                            state_dict1 = torch.load(model1_path, map_location=device)

                            print(f"Loading model 2 (Black): {model2_name}...")
                            state_dict2 = torch.load(model2_path, map_location=device)

                            # --- Get Think Time ---
                            think_time_str = input("Enter think time per move for both engines (seconds, e.g., 1.0): ")
                            try:
                                think_time = float(think_time_str)
                            except ValueError:
                                print("Invalid time, using 1.0s")
                                think_time = 1.0

                            # --- Call the game function from playGame ---
                            play_engine_vs_engine(state_dict1, state_dict2, think_time, True, False, model1_name, model2_name)
                            # --- Function call finished, loop continues ---

                        # Error handling
                        except ValueError:
                            print("Invalid input.")
                        except IndexError:
                            print("Invalid model number selected.")
                        except FileNotFoundError as e:
                            print(f"Error: Model file not found '{model1_path}' and/or '{model2_path}'.")
                        except Exception as e:
                            print(f"Unexpected error: {e}")
                            traceback.print_exc()
                            input("\nPress Enter to continue...")

                    else:
                        print("Invalid choice. Please enter 1 or 2.")

                else:
                    print("Invalid input. Please enter 1 or 2.")
        # TODO
        ''' 4. Create and save a UCI interface for the model '''
    
    
    
    