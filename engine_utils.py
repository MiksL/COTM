import os
import torch
import chess.engine
import contextlib
import math
from typing import Optional, Dict, Any, List

# Attempt to import ChessNN relative to project root
try:
    from neural_network.neuralNetwork import ChessNN
except ImportError as e:
    print(f"Warning: Could not import ChessNN in engine_utils.py: {e}. Ensure neural_network module is accessible.")
    ChessNN = None

# Load environment variables for Stockfish defaults
STOCKFISH_PATH_ENV = os.getenv("STOCKFISH_PATH")
try:
    STOCKFISH_THREADS_ENV = int(os.getenv("STOCKFISH_THREADS", "12"))
    STOCKFISH_DEFAULT_THINK_TIME_ENV = float(os.getenv("STOCKFISH_THINK_TIME", "0.1"))
except ValueError:
    print("Warning: Invalid STOCKFISH_THREADS or STOCKFISH_THINK_TIME in .env. Using hardcoded defaults.")
    STOCKFISH_THREADS_ENV = 12
    STOCKFISH_DEFAULT_THINK_TIME_ENV = 0.1


def load_chess_model(model_state_dict: Dict[str, Any], device: torch.device) -> Optional[torch.nn.Module]:
    """Loads the ChessNN model from a state dictionary."""
    if ChessNN is None:
        print("Error: ChessNN class is not available. Cannot load model.")
        return None
    try:
        model = ChessNN(input_channels=18) # Ensure constructor matches
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        print("ChessNN model loaded successfully on device.")
        return model
    except Exception as e:
        print(f"Error loading ChessNN model: {e}")
        return None

@contextlib.contextmanager
def managed_uci_engine(engine_path: Optional[str], options: Optional[Dict[str, Any]] = None, purpose: str = "engine"):
    """Provides a UCI engine within a context, ensuring it's properly closed."""
    engine = None
    if not engine_path or not os.path.exists(engine_path):
        print(f"UCI engine path '{engine_path}' for {purpose} not found or invalid. Engine not started.")
        yield None
        return
    try:
        print(f"Initializing UCI engine ({purpose}) from: {engine_path} with options: {options}")
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        if options:
            engine.configure(options)
        engine_name = engine.id.get("name", os.path.basename(engine_path))
        print(f"Managed UCI engine '{engine_name}' for {purpose} active.")
        yield engine
    except Exception as e:
        print(f"Failed to initialize/manage UCI engine '{engine_path}' for {purpose}: {e}")
        yield None
    finally:
        if engine:
            engine_name = engine.id.get("name", os.path.basename(engine_path))
            engine.quit()
            print(f"Managed UCI engine '{engine_name}' for {purpose} closed.")

def get_stockfish_options(elo_limit: Optional[int] = None, threads: Optional[int] = None) -> Dict[str, Any]:
    """Returns a dictionary of options for Stockfish."""
    sf_options = {"Threads": threads if threads is not None else STOCKFISH_THREADS_ENV}
    if elo_limit is not None:
        sf_options["UCI_LimitStrength"] = True
        sf_options["UCI_Elo"] = elo_limit
        if threads is None: # Default to 1 thread for ELO limit for consistency
             sf_options["Threads"] = 1
    return sf_options

def get_uci_eval_string(score_info: Optional[chess.engine.PovScore], board_turn: chess.Color) -> str:
    """Formats UCI score info (centipawns/mate) into a readable string from current player's POV."""
    if score_info is None:
        return "Eval: N/A"

    # Get score from the perspective of the player whose turn it is
    pov_score_val = score_info.pov(board_turn)
    
    if pov_score_val.is_mate():
        mate_in_plies = pov_score_val.mate() # Mate is in plies from current player's perspective
        mate_in_moves = math.ceil(mate_in_plies / 2) if mate_in_plies > 0 else -math.ceil(abs(mate_in_plies) / 2)
        return f"Eval: M{mate_in_moves}"
    else:
        cp_score = pov_score_val.score()
        if cp_score is not None:
            return f"Eval: {cp_score / 100.0:.2f}"
        else:
            return "Eval: N/A (cp score None)"

def get_uci_pv_san(board: chess.Board, pv_moves: Optional[List[chess.Move]]) -> str:
    """Converts a Principal Variation (PV) list of moves into a Standard Algebraic Notation (SAN) string."""
    if not pv_moves:
        return ""
    san_pv = []
    temp_board = board.copy()
    try:
        for move in pv_moves:
            if temp_board.is_legal(move):
                san_pv.append(temp_board.san(move))
                temp_board.push(move)
            else:
                break # Stop if PV contains an illegal move
    except Exception:
        pass # Ignore errors if PV is malformed or board state is unexpected
    return " PV: " + " ".join(san_pv) if san_pv else "" 