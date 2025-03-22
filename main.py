import os
import torch
import chess

from training import train_model
from playGame import ChessGameUI # TODO: Improve playGame.py - implement inference usage

from readGames import PGNReader
from dotenv import load_dotenv

import gc

if __name__ == "__main__":
    ''' 
    1. Read existing games to be used for training
    2. Turn games into usable data - board state encoding 
    '''
    
    # TODO - look at ways to improve game encoding
    # TODO - pre-encode and save games to remove redundant processing
    
    # Define the model var
    model = None
    
    # user input to train an existing game/self-play model or load a pre-trained model in the models directory
    userInput = input("1. Train from existing games\n2. Train from self-play\n3. Load a pre-trained model\n")
    if userInput.isdigit():
        if int(userInput) == 1:
            # Train from existing games
            load_dotenv()
            PGN_PATH = os.getenv("PGN_PATH") # Get PGN path from environment variable - points to game file
            sampler = PGNReader(PGN_PATH)
            
            positions, moves, values = sampler.preprocess_games(80000, 12)
            
            # Close the file, release resources - sampler not needed after game pre-processing
            del sampler
            gc.collect()
            
            ''' 
            3. Train the models
            3.1. Learns from the games generated/given
            3.2. Learns from self-play
            '''
            # 3.1
            # Train the model
            model = train_model(
                positions=positions,
                moves=moves,
                values=values,
                epochs=20,
                batch_size=2048,
                learning_rate=1e-3,
                val_split=0.1,
            )
            
            del positions, moves, values
            gc.collect()
            
            saveModel = input("Save the model? (y/n)")
            if saveModel.lower() == 'y':
                # Save the model in the models directory
                torch.save(model.state_dict(), "models/TestBotEG.pth")
                pass
            else:
                pass

        elif int(userInput) == 2:
            # Train from self-play
            print("Training from self-play")
        elif int(userInput) == 3:
            # Loads all the model names from the models directory, prompting the user to select one
            models = [model for model in os.listdir("models") if model.endswith(".pth")]
            print("Select a model to load:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
                modelIndex = int(input())
                
                # Load the user-selected model
                model = torch.load(f"models/{models[modelIndex-1]}")
                
                # Check if model is loaded - debug
                # Output name of model
                print(f"Model loaded: {models[modelIndex-1]}")
                
                # Inference test with a random board
                #board = chess.Board()
                #policy_probs, value = predict_moves(model, [board])
                #print(f"Position evaluation: {value[0][0]:.4f}")
                
                # Play game TODO - Implement updated playGame.py
                #game = ChessGame(model, think_time=1.0)
                #game.play()
                
                game = ChessGameUI(model, think_time=1.0)
                
                while not game.is_game_over():
                    print(game.board)
                    
                    if game.board.turn == chess.WHITE:
                        # Handle player move via UI
                        move_uci = input("Your move (UCI) or 'undo' to take back: ")
                        
                        if move_uci.lower() == 'undo':
                            if game.undo_move():
                                # Undo again to get back to player's turn
                                game.undo_move()
                                print("Move undone")
                            else:
                                print("Cannot undo!")
                            continue
                        elif move_uci.lower() == 'reset':
                            game.reset()
                            print("Game reset")
                            continue
                            
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if game.make_player_move(move):
                                print("Move made")
                            else:
                                print("Illegal move!")
                        except ValueError:
                            print("Invalid format!")
                    else:
                        # Engine move
                        game.make_engine_move()
                
                print(f"Game over: {game.get_result()}")
        else:
            print("Invalid input")
    
    # TODO
    ''' 4. Create and save a UCI interface for the model '''
    
    
    
    