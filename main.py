import os
import torch
import chess

from training import train_model
#from playGame import ChessGame - TODO: Implement playGame.py for updated neuralNetwork

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
    
    load_dotenv()
    PGN_PATH = os.getenv("PGN_PATH") # Get PGN path from environment variable - points to game file
    sampler = PGNReader(PGN_PATH)
    
    positions, moves, values = sampler.preprocess_games(12000, 12)
    
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
    
    # Save the model
    torch.save(model.state_dict(), "chessBot.pth")
    
    # Example of inference with the trained model
    board = chess.Board()
    #policy_probs, value = predict_moves(model, [board])
    #print(f"Position evaluation: {value[0][0]:.4f}")
    
    # Play game TODO - Implement updated playGame.py
    #game = ChessGame(model, think_time=1.0)
    #game.play()
    
    # TODO
    ''' 4. Create and save a UCI interface for the model '''
    
    
    
    