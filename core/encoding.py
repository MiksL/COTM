import chess
import numpy as np

class ChessEncoder:
    """
    Encoding class for board and move encoding using a precomputed move map.
    Generates only potentially legal moves respecting piece movement rules.
    """
    def __init__(self):
        self.all_possible_moves = []
        self.move_to_index = {}
        self._build_move_maps()
        self.num_possible_moves = len(self.all_possible_moves)
        # Expected size = 1968 moves
        print(f"Generated {self.num_possible_moves} unique potentially legal move objects.")
        
    PIECE_VALUES = {
        chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
        chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
    }

    def _add_move(self, move_set, move):
        move_set.add(move)

    def _build_move_maps(self):
        print("Building move maps (piece-based encoding)")
        move_set = set()
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT] # Default promo = Queen

        # Iterate over all board squares
        for from_sq in range(64):
            from_rank = chess.square_rank(from_sq)
            from_file = chess.square_file(from_sq)

            # --- Pawn move generation ---

            # White Pawn check
            if from_rank < 7: # Pawns cannot start on the 8th rank
                # Moving forward 1 square
                to_sq_f1 = from_sq + 8
                if chess.square_rank(to_sq_f1) < 8: # Check if still within board limits
                    if from_rank == 6: # Moving to rank 7 - must promote
                        for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_f1, promotion=p))
                    else: # Normal move
                        self._add_move(move_set, chess.Move(from_sq, to_sq_f1))

                # Forward 2 squares (only possible if starting on rank 1)
                if from_rank == 1:
                    to_sq_f2 = from_sq + 16
                    # No need to check if within board limits
                    self._add_move(move_set, chess.Move(from_sq, to_sq_f2))

                # Captures with board edge checks
                # Capture Left (diagonal up-left)
                if from_file > 0:
                    to_sq_cl = from_sq + 7
                    if chess.square_rank(to_sq_cl) < 8: # Check if still within board limits
                         if from_rank == 6: # Moving to rank 7 - must promote
                             for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_cl, promotion=p))
                         else: # Normal capture
                             self._add_move(move_set, chess.Move(from_sq, to_sq_cl))
                # Capture Right (diagonal up-right)
                if from_file < 7:
                    to_sq_cr = from_sq + 9
                    if chess.square_rank(to_sq_cr) < 8: # Check if still within board limits
                         if from_rank == 6: #  Moving to rank 7 - must promote
                             for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_cr, promotion=p))
                         else: # Normal capture
                             self._add_move(move_set, chess.Move(from_sq, to_sq_cr))

            # Black pawn check
            if from_rank > 0: # Pawns cannot start on the 1st rank
                # Moving forward 1 square
                to_sq_f1 = from_sq - 8
                if chess.square_rank(to_sq_f1) >= 0: # Check if still within board limits
                    if from_rank == 1: # Moving to rank 0 - must promote
                        for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_f1, promotion=p))
                    else: # Normal move
                        self._add_move(move_set, chess.Move(from_sq, to_sq_f1))

                # Forward 2 squares (only possible if starting on rank 6)
                if from_rank == 6:
                    to_sq_f2 = from_sq - 16
                    # No need to check if within board limits
                    self._add_move(move_set, chess.Move(from_sq, to_sq_f2))

                # Captures with board edge checks
                # Capture Left (diagonal down-left relative to board)
                if from_file > 0:
                    to_sq_cl = from_sq - 9
                    if chess.square_rank(to_sq_cl) >= 0: # Check if still within board limits
                        if from_rank == 1: # Must promote
                            for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_cl, promotion=p))
                        else: # Normal capture
                             self._add_move(move_set, chess.Move(from_sq, to_sq_cl))
                # Capture Right (diagonal down-right relative to board)
                if from_file < 7:
                    to_sq_cr = from_sq - 7
                    if chess.square_rank(to_sq_cr) >= 0: # Check if still within board limits
                        if from_rank == 1: # Must promote
                            for p in promotion_pieces: self._add_move(move_set, chess.Move(from_sq, to_sq_cr, promotion=p))
                        else: # Normal capture
                             self._add_move(move_set, chess.Move(from_sq, to_sq_cr))

            # --- Generate Knight Moves ---
            knight_moves = [(1, 2), (1, -2), (-1, 2), (-1, -2),
                            (2, 1), (2, -1), (-2, 1), (-2, -1)]
            # 8 possible knight moves - L shaped
            for dr, df in knight_moves:
                to_rank, to_file = from_rank + dr, from_file + df
                if 0 <= to_rank < 8 and 0 <= to_file < 8: # Check if within board limits
                    to_sq = chess.square(to_file, to_rank)
                    self._add_move(move_set, chess.Move(from_sq, to_sq))

            # --- Generate Sliding Piece Moves (Bishop/Rook/Queen) ---
            # Directions: N, NE, E, SE, S, SW, W, NW
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            for dr, df in directions:
                for step in range(1, 8): # Max 7 steps
                    to_rank, to_file = from_rank + dr * step, from_file + df * step
                    if 0 <= to_rank < 8 and 0 <= to_file < 8: # Check bounds
                        to_sq = chess.square(to_file, to_rank)
                        # Add move - set handles overlap (Queen=Bishop+Rook)
                        self._add_move(move_set, chess.Move(from_sq, to_sq))
                    else:
                        break # Stop sliding in this direction if off board

            # King non-castling moves
            for dr, df in directions: # Use all 8 directions for step=1
                 to_rank, to_file = from_rank + dr, from_file + df
                 if 0 <= to_rank < 8 and 0 <= to_file < 8: # Check bounds
                     to_sq = chess.square(to_file, to_rank)
                     self._add_move(move_set, chess.Move(from_sq, to_sq))

        # Castling
        # King E1 moves
        self._add_move(move_set, chess.Move(chess.E1, chess.G1)) # White O-O
        self._add_move(move_set, chess.Move(chess.E1, chess.C1)) # White O-O-O
        # King E8 moves
        self._add_move(move_set, chess.Move(chess.E8, chess.G8)) # Black O-O
        self._add_move(move_set, chess.Move(chess.E8, chess.C8)) # Black O-O-O

        # Sort for consistency
        self.all_possible_moves = sorted(list(move_set), key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
        self.move_to_index = {move: idx for idx, move in enumerate(self.all_possible_moves)}

    def encode_move(self, move: chess.Move) -> int:
        """Encodes a chess.Move object into its unique index."""
        encoded_index = self.move_to_index.get(move)
        if encoded_index is None:
            print(f"Warning: Move {move.uci()} not found in precomputed map!")
            return -1 # Error if move is invalid
        return encoded_index

    def decode_move(self, index: int) -> chess.Move | None:
        """Decodes an index back into a chess.Move object."""
        if 0 <= index < self.num_possible_moves:
            return self.all_possible_moves[index]
        else:
            print(f"Warning: Move index {index} out of range ({self.num_possible_moves} total)!")
            return None # No move found
        
    @staticmethod
    def encode_board(board):
        # Allocate planes
        planes = np.zeros((18, 8, 8), dtype=np.uint8)

        # Initialize material counters
        white_material = 0.0
        black_material = 0.0

        # Iterate piece map
        piece_map_items = board.piece_map().items() # Get items

        for square, piece in piece_map_items:
            # 1. Place piece on corresponding plane
            rank, file = divmod(square, 8)
            piece_type = piece.piece_type
            piece_idx = piece_type - 1  # 0-5

            # 2. Accumulate material values for balance plane
            piece_value = ChessEncoder.PIECE_VALUES.get(piece_type, 0.0)

            if piece.color == chess.WHITE:
                planes[piece_idx, rank, file] = 1
                white_material += piece_value
            else: # piece.color == chess.BLACK
                planes[piece_idx + 6, rank, file] = 1 # Offset index for black
                black_material += piece_value

        # Turn plane
        if board.turn == chess.WHITE:
            planes[12].fill(1)

        # Castling planes
        if board.has_kingside_castling_rights(chess.WHITE): planes[13].fill(1)
        if board.has_queenside_castling_rights(chess.WHITE): planes[14].fill(1)
        if board.has_kingside_castling_rights(chess.BLACK): planes[15].fill(1)
        if board.has_queenside_castling_rights(chess.BLACK): planes[16].fill(1)

        # Material balance plane calculation
        total_material = white_material + black_material
        if total_material > 0:
            # Calculate ratio as float, then scale and cast to int for uint8 plane
            white_ratio_scaled = int((white_material / total_material) * 255.0)
             # Value is clamped to [0, 255]
            planes[17].fill(max(0, min(255, white_ratio_scaled)))
        else:
            planes[17].fill(128) # If no material - neutral value

        return planes
    
    @staticmethod
    def decode_board(planes: np.ndarray) -> chess.Board:
        """Decodes a board from a 18x8x8 numpy array. Only decodes piece positions - used for testing"""
        board = chess.Board(None)
        piece_map = board.piece_map()
        
        # Planes 1-12 are the only relevant ones to decode state back into a board
        # 1-6 - white pieces
        # 7-12 - black pieces
        
        # Iterate over the planes, place pieces accordingly
        for piece_idx in range(6):
            # Get all squares with this piece type
            squares = np.argwhere(planes[piece_idx] == 1)
            for rank, file in squares:
                piece_type = piece_idx + 1 # Convert to 1-7
                piece = chess.Piece(piece_type, chess.WHITE)
                piece_map[chess.square(file, rank)] = piece
        
        # Iterate over the black planes
        for piece_idx in range(6):
            # Get all squares with this piece type
            squares = np.argwhere(planes[piece_idx + 6] == 1)
            for rank, file in squares:
                piece_type = piece_idx + 1 # Convert to 1-7
                piece = chess.Piece(piece_type, chess.BLACK)
                piece_map[chess.square(file, rank)] = piece
                
                
        # Turn plane decoding (shows next player to move)
        if planes[12].sum() > 0:
            board.turn = chess.WHITE
        else:
            board.turn = chess.BLACK
        
                
        # TODO - if deemed necessary, add other relevant plane decoding
        
        
        
        # Update the board with the new piece map
        board.set_piece_map(piece_map)
        return board
        
        
        
        
