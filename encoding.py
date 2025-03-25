import numpy as np
import chess

class ChessEncoder:
    """ Encoding class for board and move encoding """
    
    @staticmethod
    def encode_board(board):
        """Encode chess board as planes of features"""
        
        # 18 planes for each feature in 8x8 grid
        planes = np.zeros((18, 8, 8), dtype=np.uint8)

        # Planes 0-11: Piece positions (6 piece types × 2 colors)
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_idx = piece.piece_type - 1
            if piece.color == chess.BLACK:
                piece_idx += 6
            planes[piece_idx, rank, file] = 1

        # Plane 12: Side to move
        if board.turn == chess.WHITE:
            planes[12].fill(1)

        # Planes 13-16: Castling rights
        planes[13].fill(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        planes[14].fill(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        planes[15].fill(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        planes[16].fill(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)

        # Plane 17: Material balance (scaled to 0-255)
        piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
            chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
        }

        white_material = sum(piece_values[p.piece_type] for sq, p in board.piece_map().items() 
                            if p.color == chess.WHITE)
        black_material = sum(piece_values[p.piece_type] for sq, p in board.piece_map().items() 
                            if p.color == chess.BLACK)

        total_material = white_material + black_material
        if total_material > 0:
            white_ratio = white_material / total_material
            planes[17].fill(int(white_ratio * 255))
        else:
            planes[17].fill(128)  # Equal material

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
    def decode_move(move_idx):
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