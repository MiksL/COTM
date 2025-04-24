import numpy as np
import chess

class ChessEncoder:
    """ Encoding class for board and move encoding """
    
    PIECE_VALUES = {
        chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
        chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
    }
    
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
            # Calculate ratio as float, then scaled and casted to int for uint8 plane
            white_ratio_scaled = int((white_material / total_material) * 255.0)
             # Value is clamped to [0, 255] just in case
            planes[17].fill(max(0, min(255, white_ratio_scaled)))
        else:
            planes[17].fill(128) # If no material - neutral value

        return planes

    @staticmethod
    def encode_move(move):
        """Encode chess move as a unique index"""
        from_square = move.from_square
        to_square = move.to_square
        base_idx = from_square * 69 # 64 possible squares + 5 promotion options - 69 total options
        
        if move.promotion is None:
            return base_idx + to_square
        
        # Offset for promotion handling
        promo_offset = {
            chess.QUEEN: 64, chess.ROOK: 65, chess.BISHOP: 66,
            chess.KNIGHT: 67, chess.PAWN: 68
        }
        return base_idx + promo_offset[move.promotion]
    
    @staticmethod
    def decode_move(move_idx): # TODO - fix incorrect decoding(?)
        """Decode a move index back to a chess.Move object."""
        from_square = move_idx // 69
        to_square = move_idx % 69

        if to_square < 64:
            # Regular move
            return chess.Move(from_square, to_square)
        else:
            # Promotion move
            promotion_map = {64: chess.QUEEN, 65: chess.ROOK, 66: chess.BISHOP, 67: chess.KNIGHT, 68: chess.PAWN}
            promotion_piece = promotion_map[to_square]
            return chess.Move(from_square, to_square - 64, promotion=promotion_piece)