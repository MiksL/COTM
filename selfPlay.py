import chess
import torch.nn.functional as F
import numpy as np
import concurrent.futures
import threading
import os
from tqdm import tqdm
from dataset import ChessDataset
from gpuInference import GPUInferenceServer

class SelfPlayChess:
    def __init__(self, model, encoder, num_games=1000, temperature=1.0, batch_size=32, num_workers=None):
        self.model = model
        self.encoder = encoder
        self.num_games = num_games
        self.temperature = temperature
        
        # Auto-determine workers if not specified
        if num_workers is None:
            num_workers = min(12, os.cpu_count())
        self.num_workers = num_workers
        
        # Thread-safe storage for game data
        self.lock = threading.Lock()
        self.positions = []
        self.moves = []
        self.values = []
        
        # For calculating metrics
        self.total_positions = 0
        self.total_games = 0
        
        # Create GPU inference server
        self.inference_server = GPUInferenceServer(model, batch_size=batch_size)
        
        # Pre-compute move lookup for more efficient decoding
        self.move_map = self._build_move_map()
    
    def _build_move_map(self):
        """Build a lookup table for move encoding/decoding"""
        move_map = {}
        for from_square in range(64):
            for to_square in range(64):
                # Regular moves
                move = chess.Move(from_square, to_square)
                idx = self.encoder.encode_move(move)
                move_map[idx] = move
                
                # Promotion moves (only from ranks 2 and 7)
                if from_square // 8 == 1 or from_square // 8 == 6:
                    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        promotion = chess.Move(from_square, to_square, promotion=piece_type)
                        idx = self.encoder.encode_move(promotion)
                        move_map[idx] = promotion
        return move_map
    
    def simulate_game(self):
        """Simulate a single game with improved efficiency"""
        game_positions = []
        game_moves = []
        game_values = []
        
        board = chess.Board()
        
        while not board.is_game_over():
            # Encode the current board state
            encoded_board = self.encoder.encode_board(board)
            game_positions.append(encoded_board)
            
            # Submit to inference server and get result
            try:
                policy, value = self.inference_server.submit(encoded_board)
            except Exception as e:
                # Fallback to random move
                legal_moves = list(board.legal_moves)
                move = np.random.choice(legal_moves)
                move_idx = self.encoder.encode_move(move)
                game_moves.append(move_idx)
                board.push(move)
                continue
            
            # Apply temperature scaling for exploration
            policy = F.softmax(policy / self.temperature, dim=0)
            
            # Filter out illegal moves (more efficient direct indexing)
            legal_moves = list(board.legal_moves)
            legal_move_indices = [self.encoder.encode_move(move) for move in legal_moves]
            
            try:
                legal_policy = policy[legal_move_indices]
                # Normalize the probabilities of legal moves
                legal_policy = legal_policy / legal_policy.sum()
                
                # Sample a move from the legal moves based on the policy
                move_idx_index = np.random.choice(len(legal_move_indices), p=legal_policy.numpy())
                move_idx = legal_move_indices[move_idx_index]
                move = self.move_map.get(move_idx)
                
                # Fallback if move not in map
                if move not in legal_moves:
                    move = legal_moves[move_idx_index]
                    move_idx = self.encoder.encode_move(move)
            except Exception as e:
                # Fallback to random move if sampling fails
                move = np.random.choice(legal_moves)
                move_idx = self.encoder.encode_move(move)
            
            game_moves.append(move_idx)
            board.push(move)
        
        # Get final result and create values
        result = board.result()
        final_value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result]
        game_values = [final_value * ((-1) ** i) for i in range(len(game_positions))]
        
        # Thread-safe update of our data structures
        with self.lock:
            self.positions.extend(game_positions)
            self.moves.extend(game_moves)
            self.values.extend(game_values)
            self.total_positions += len(game_positions)
            self.total_games += 1
        
        return len(game_positions)  # Return number of positions for stats
    
    def run_self_play(self):
        """Run self-play games using multithreading"""
        print(f"Using {self.num_workers} worker threads")
        
        try:
            # Use multi-threading to simulate games in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit games in batches to avoid overwhelming the queue
                pending_futures = set()
                
                # Submit initial batch of games
                initial_batch = min(self.num_workers * 2, self.num_games)
                for _ in range(initial_batch):
                    future = executor.submit(self.simulate_game)
                    pending_futures.add(future)
                
                completed_games = 0
                
                # Process completed games and submit new ones
                with tqdm(total=self.num_games, desc="Simulating Games") as pbar:
                    while pending_futures:
                        # Wait for a game to complete
                        done, pending_futures = concurrent.futures.wait(
                            pending_futures,
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Process completed games
                        for future in done:
                            try:
                                positions_added = future.result()
                                completed_games += 1
                                pbar.update(1)
                            except Exception as e:
                                print(f"Game simulation failed: {e}")
                        
                        # Submit new games if needed
                        games_to_submit = min(len(done), self.num_games - completed_games - len(pending_futures))
                        for _ in range(games_to_submit):
                            future = executor.submit(self.simulate_game)
                            pending_futures.add(future)
                        
                        # Update progress bar with additional info
                        avg_positions = self.total_positions / max(1, self.total_games)
                        pbar.set_postfix({
                            'positions': self.total_positions,
                            'avg_len': f'{avg_positions:.1f}'
                        })
                        
                        # Stop if we've completed all games
                        if completed_games >= self.num_games:
                            break
            
            print(f"Generated {self.total_positions} positions from {self.total_games} games")
        
        finally:
            # Ensure inference server is shut down properly
            self.inference_server.shutdown()
        
        # Convert to numpy arrays
        positions = np.array(self.positions)
        moves = np.array(self.moves)
        values = np.array(self.values).reshape(-1, 1)
        
        return ChessDataset(positions, moves, values)