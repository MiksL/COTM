import chess
import chess.pgn
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import concurrent.futures
import gc

from core.encoding import ChessEncoder

class PGNReader:
    """Reader for chess PGN files with preprocessing"""
    
    def __init__(self, pgn_path):
        """Initialize with path to PGN file"""
        self.pgn_path = pgn_path
        # Move/position encoding class
        self.encoder = ChessEncoder()
        
    def read_games(self, games=None, num_workers=1):
        """Read games from PGN file using sequential or parallel processing, depending on assigned num_workers (thread count)"""
        # Handle sequential case with simple iterator
        if num_workers <= 1:
            game_list = []
            with open(self.pgn_path) as pgn:
                iterator = tqdm(range(games)) if games is not None else iter(lambda: True, False)
                for _ in iterator:
                    game = chess.pgn.read_game(pgn)
                    if game is None:  # End of file reached
                        break
                    moves = list(game.mainline_moves())
                    if moves:  # Only append games with moves
                        game_list.append(moves)
                    if games is not None and len(game_list) >= games:
                        break
            return game_list
            
        # For parallel processing files are split into chunks
        file_size = os.path.getsize(self.pgn_path)
        chunk_size = file_size // num_workers
        chunk_positions = [i * chunk_size for i in range(num_workers)] + [file_size] # Chunk positions for each worker (where to start looking for assigned games)
        
        # Worker argument preparation
        worker_args = []
        for i in range(num_workers):
            worker_games = None # Assigned game count for each worker
            if games is not None:
                worker_games = games // num_workers # Games / workers - games per worker
                if i == num_workers - 1: # Last worker gets remaining games
                    worker_games += games % num_workers
            worker_args.append((self.pgn_path, chunk_positions[i], chunk_positions[i+1], worker_games))
        
        # Process chunks with multiprocessing.Pool
        all_games = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(self._read_games_chunk, worker_args)
            for game_list in results:
                all_games.extend(game_list)
                
        # Limit the number of games if limited
        if games is not None and len(all_games) > games:
            all_games = all_games[:games]
            
        return all_games
    
    def _read_games_chunk(self, pgn_path, start_pos, end_pos, games=None):
        """Read a specific chunk of the PGN file"""
        game_list = []
        
        with open(pgn_path, 'r') as pgn:
            pgn.seek(start_pos)
            
            # Skip to the next game if starting at a different position
            if start_pos > 0:
                line = pgn.readline()
                while line and not line.startswith('[Event '):
                    line = pgn.readline()
            
            # Read games until the end of the chunk or game limit is reached
            games_read = 0
            while (pgn.tell() < end_pos) and (games is None or games_read < games):
                game = chess.pgn.read_game(pgn)
                if game is None:  # End of file
                    break
                    
                moves = list(game.mainline_moves()) # Only read mainline moves that were played
                if moves:
                    game_list.append(moves)
                    games_read += 1
        
        return game_list
    
    def preprocess_games(self, games, num_workers=1):
        """
        Read and preprocess games
        
        Args:
            games: number of games to read
            num_workers: number of workers (cpu threads) to use for game preprocessing
        """
        # Read the games from the PGN file
        print(f"Reading {games if games is not None else 'all'} games from {self.pgn_path}")
        game_list = self.read_games(games, num_workers=num_workers)
        print(f"Read {len(game_list)} games")
        
        # Processed data temp storage
        all_positions, all_moves, all_values = [], [], []
        
        # Batch creation if using multiple workers
        if num_workers > 1:
            batch_size = max(1, len(game_list) // num_workers) # Batch size - length of game list divided by number of workers
            game_batches = [game_list[i:i + batch_size] for i in range(0, len(game_list), batch_size)] # Games split into batches
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor: # ProcessPoolExecutor parallel processing
                futures = [executor.submit(self._process_game_batch, batch) for batch in game_batches]
                
                for future in tqdm(concurrent.futures.as_completed(futures), # tqdm progress bar for game batch processing TODO - rework or maybe remove, not giving too much insight now
                                  total=len(futures), 
                                  desc="Processing game batches"):
                    try:
                        positions, moves, values = future.result()
                        all_positions.extend(positions)
                        all_moves.extend(moves)
                        all_values.extend(values)
                    except Exception as e:
                        print(f"Error processing game batch: {e}")
        else:
            # Process games sequentially if one worker is assigned
            positions, moves, values = self._process_game_batch(game_list)
            all_positions.extend(positions)
            all_moves.extend(moves)
            all_values.extend(values)
        
        # Convert to numpy arrays
        positions_array = np.array(all_positions, dtype=np.uint8)
        move_indices_array = np.array(all_moves, dtype=np.int16)
        values_array = np.array(all_values, dtype=np.float16).reshape(-1, 1)
        
        # Delete temporary data, garbage collect
        del all_positions, all_moves, all_values, game_list
        gc.collect()
        
        print(f"Preprocessed {len(positions_array)} positions")
        
        return positions_array, move_indices_array, values_array
    
    def _process_game_batch(self, game_batch):
        """Process a batch of games"""
        batch_positions, batch_moves, batch_values = [], [], []
        
        # For each game, encode board and moves
        for game in game_batch:
            board = chess.Board()
            for move in game:
                batch_positions.append(self.encoder.encode_board(board))
                batch_moves.append(self.encoder.encode_move(move))
                
                # TODO - Implement a better evaluation function - find games with pre-evaluated positions(?)
                batch_values.append(0.0)  # Default value - no evaluation for engine, move prediction only
                
                board.push(move)
        
        return batch_positions, batch_moves, batch_values