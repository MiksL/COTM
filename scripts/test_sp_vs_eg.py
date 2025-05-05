import torch
import os
import traceback
import chess
import time
from core.encoding import ChessEncoder
from core.encodingOld import ChessEncoder as ChessEncoderOld
from neural_network.neuralNetworkSP import MCTSChessNN
from neural_network.neuralNetwork import ChessNN
from core.mcts.mcts import MCTS  # Import MCTS engine directly
import chess.pgn

# Model paths and simulation config
SP_MODEL_PATH = "models/MCTS_SelfPlay_Bot.pth"
EG_MODEL_PATH = "models/newModel-v2-2mil-ep17of25.pth"
NUM_SIMULATIONS = 50

# Helper method to load models
def load_model_from_file(model_class, policy_size, model_path):
    """Helper function to instantiate and load a model."""
    print(f"Loading {model_path} using {model_class.__name__}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate correct class with correct parameters
    if model_class == MCTSChessNN:
         model = MCTSChessNN(policy_output_size=policy_size)
    elif model_class == ChessNN:
         model = ChessNN()
    else:
         raise ValueError(f"Unknown model class: {model_class}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully to {device}.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}: {e}")
        traceback.print_exc()
        return None

# Engine vs engine play function
def custom_play_engine_vs_engine(model1, model2, encoder1, encoder2, num_simulations, 
                                 model1_name="Model1", model2_name="Model2",
                                 pause_after_move=False):
    """Custom function to play SP model vs EG model with different encoders."""
    stockfish = None  # No stockfish for simplicity
    save_dir = "saved_engine_games"; os.makedirs(save_dir, exist_ok=True)
    exploration_weight = 1.4  # Default PUCT exploration parameter
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nEngines using device: {device}")
        
        # Setup game
        board = chess.Board()  # Create initial board
        print("\n--- Starting Engine vs Engine Game ---")
        print(f"Engine 1 (White): {model1_name}")
        print(f"Engine 2 (Black): {model2_name}")
        print(f"MCTS Simulations: {num_simulations} per move\n")
        
        # Setup PGN
        game_pgn = chess.pgn.Game()
        game_pgn.headers["Event"] = "SP vs EG Test Game"
        game_pgn.headers["Site"] = "Bachelor Testing"
        game_pgn.headers["Date"] = time.strftime("%Y.%m.%d")
        game_pgn.headers["White"] = model1_name
        game_pgn.headers["Black"] = model2_name
        game_pgn.headers["Round"] = "1"
        current_pgn_node = game_pgn
        
        # Game loop
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            print(f"\n--- Move {move_count} ---")
            print(board)
            
            # Create a fresh MCTS instance for the current position
            if board.turn == chess.WHITE:
                # Create engine1 for White with the current board
                current_engine = MCTS(
                    model=model1, 
                    encoder=encoder1, 
                    initial_board=board.copy(),  # Pass a copy of the current board
                    simulations=num_simulations,
                    exploration_weight=exploration_weight  # Add exploration weight
                )
                engine_name = f"Engine 1 (W - {model1_name})"
            else:
                # Create engine2 for Black with the current board
                current_engine = MCTS(
                    model=model2, 
                    encoder=encoder2, 
                    initial_board=board.copy(),  # Pass a copy of the current board
                    simulations=num_simulations,
                    exploration_weight=exploration_weight  # Add exploration weight
                )
                engine_name = f"Engine 2 (B - {model2_name})"
            
            print(f"{engine_name} thinking (MCTS)...")
            best_move, move_visits, q_stats = current_engine.get_best_move_and_stats()  # Note we now get 3 return values
            
            if best_move:
                print(f"{engine_name} plays: {best_move.uci()}")
                board.push(best_move)
                current_pgn_node = current_pgn_node.add_variation(best_move)
                if pause_after_move:
                    input("Press Enter to continue...")
            else:
                print(f"{engine_name} failed to find a legal move!")
                break
        
        # Game end
        print("\n--- Game Over ---")
        print(board)
        print(f"Result: {board.result()}")
        
        # Save PGN
        game_pgn.headers["Result"] = board.result()
        game_time = time.strftime("%Y%m%d-%H%M%S")
        pgn_file = os.path.join(save_dir, f"SP_vs_EG_{game_time}.pgn")
        with open(pgn_file, "w") as f:
            f.write(str(game_pgn))
        print(f"Game saved to {pgn_file}")
        
    except Exception as e:
        print(f"\nError during game: {e}")
        traceback.print_exc()

# Main test function
if __name__ == "__main__":
    print("--- Self-Play vs Existing-Games Model Test Tool ---")
    print("1. Human vs Self-Play Model")
    print("2. Human vs Existing-Games Model")
    print("3. Self-Play vs Existing-Games Model (computer vs computer)")
    choice = input("Select option (1-3): ")
    
    # Check model paths
    if not os.path.exists(SP_MODEL_PATH):
        print(f"Error: Self-Play model path not found: {SP_MODEL_PATH}")
        exit()
    if not os.path.exists(EG_MODEL_PATH):
        print(f"Error: Existing-Games model path not found: {EG_MODEL_PATH}")
        exit()
    
    # Initialize encoders with correct sizes
    try:
        new_encoder = ChessEncoder()   # 1968 moves for SP model
        old_encoder = ChessEncoderOld() # Need to instantiate this for EG model
        
        # For SP model
        sp_policy_size = new_encoder.num_possible_moves  # Should be 1968
        # For EG model - we don't need to since ChessNN doesn't take a parameter
    except Exception as e:
        print(f"Error initializing encoders: {e}")
        exit()
    
    try:
        if choice == "3":
            # Computer vs Computer: SP vs EG
            print("\nSetting up Self-Play vs Existing-Games match...")
            
            # Load models with their correct encoders
            sp_model = load_model_from_file(MCTSChessNN, sp_policy_size, SP_MODEL_PATH)
            eg_model = load_model_from_file(ChessNN, None, EG_MODEL_PATH)
            
            if sp_model and eg_model:
                # Ask for simulation count
                try:
                    sim_input = input("Enter MCTS simulation count (default 50): ").strip()
                    simulations = int(sim_input) if sim_input else NUM_SIMULATIONS
                except ValueError:
                    simulations = NUM_SIMULATIONS
                
                # Decide who plays white
                sp_plays_white = input("Should Self-Play model play as White? (y/n): ").lower() == 'y'
                pause = input("Pause after each move? (y/n): ").lower() == 'y'
                
                print("\nStarting game...")
                if sp_plays_white:
                    custom_play_engine_vs_engine(
                        model1=sp_model,
                        model2=eg_model,
                        encoder1=new_encoder,  # SP model needs new encoder
                        encoder2=old_encoder, # EG model needs old encoder
                        num_simulations=simulations,
                        model1_name=f"SP_{os.path.basename(SP_MODEL_PATH)}",
                        model2_name=f"EG_{os.path.basename(EG_MODEL_PATH)}",
                        pause_after_move=pause
                    )
                else:
                    custom_play_engine_vs_engine(
                        model1=eg_model,
                        model2=sp_model,
                        encoder1=old_encoder, # EG model needs old encoder
                        encoder2=new_encoder,  # SP model needs new encoder
                        num_simulations=simulations,
                        model1_name=f"EG_{os.path.basename(EG_MODEL_PATH)}",
                        model2_name=f"SP_{os.path.basename(SP_MODEL_PATH)}",
                        pause_after_move=pause
                    )
            else:
                print("\nCould not start game because one or both models failed to load.")
        else:
            print("This option is not yet implemented.")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred:")
        traceback.print_exc()
    
    print("\n--- Test finished ---")