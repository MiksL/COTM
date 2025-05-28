import chess
import random
import numpy as np
import torch
from stockfish import Stockfish
from tqdm import tqdm
from dotenv import load_dotenv
import os
import statistics
from collections import defaultdict
from neural_network.neuralNetwork import ChessNN

# Benchmark class implementation to test model against stockfish engine moves
class Benchmark:
    def __init__(self, model_dict_1, encoder, model_dict_2=None, depth=10):
        """
        Compare one or two neural network models against Stockfish.
        
        Args:
            model_dict_1: First model state_dict to evaluate.
            encoder: Encoder instance for board representation.
            model_dict_2: Optional second model state_dict.
            depth: Stockfish analysis depth.
        """
        load_dotenv()
        stockfish_path = os.getenv("STOCKFISH_PATH")
        if not stockfish_path:
            raise ValueError("STOCKFISH_PATH environment variable not set")
        
        device = torch.device("cpu")
        
        self.model_1 = ChessNN(input_channels=18)
        self.model_1.load_state_dict(model_dict_1)
        self.model_1.to(device)
        self.model_1.eval()
        
        self.model_2 = None
        if model_dict_2 is not None:
            self.model_2 = ChessNN(input_channels=18)
            self.model_2.load_state_dict(model_dict_2)
            self.model_2.to(device)
            self.model_2.eval()
        
        self.encoder = encoder
        
        try:
            self.stockfish = Stockfish(path=stockfish_path, depth=depth)
            print(f"Stockfish engine initialized (Depth: {depth})")
        except Exception as e:
            raise Exception(f"Failed to initialize Stockfish: {e}")
    
    def generate_random_position(self, min_moves=10, max_moves=40):
        """Generate a random chess position by making a random number of legal moves."""
        board = chess.Board()
        num_moves = random.randint(min_moves, max_moves)
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        return board
    
    def get_model_move_and_top_k(self, model, board, k=3):
        """Get the top k predicted moves from the model with their probabilities."""
        encoded_board = self.encoder.encode_board(board)
        board_tensor = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = model(board_tensor)
        
        policy_probs = torch.softmax(policy, dim=1).squeeze().numpy()
        legal_moves = list(board.legal_moves)
        
        move_probs = []
        for move in legal_moves:
            move_idx = self.encoder.encode_move(move)
            if 0 <= move_idx < len(policy_probs):
                 move_probs.append((move, policy_probs[move_idx]))

        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        if not move_probs:
            return None, []
            
        return move_probs[0][0] if move_probs else None, move_probs[:k]
    
    def get_stockfish_evaluation(self, fen, move_to_make: chess.Move):
        """Get Stockfish's centipawn evaluation after a specific move is made on the FEN."""
        self.stockfish.set_fen_position(fen)
        move_str = move_to_make.uci()
        
        if self.stockfish.is_move_correct(move_str):
            self.stockfish.make_moves_from_current_position([move_str])
            evaluation = self.stockfish.get_evaluation()
            
            if evaluation['type'] == 'mate':
                mate_in = evaluation['value']
                return (10000 - abs(mate_in) * 100) if mate_in > 0 else (-10000 + abs(mate_in) * 100)
            return evaluation['value']
        return 0
    
    def analyze_position(self, board, model_top_k_moves, stockfish_best_move):
        """Analyze model's top k moves against Stockfish's best move for a given position."""
        fen = board.fen()
        
        eval_after_stockfish_move = self.get_stockfish_evaluation(fen, stockfish_best_move)

        analysis = {
            'fen': fen,
            'stockfish_move': {
                'move': stockfish_best_move.uci(),
                'eval_after_move': eval_after_stockfish_move 
            },
            'model_moves_analysis': []
        }
        
        for model_move, model_prob in model_top_k_moves:
            eval_after_model_move = self.get_stockfish_evaluation(fen, model_move)
            cp_loss = eval_after_stockfish_move - eval_after_model_move
            
            analysis['model_moves_analysis'].append({
                'move': model_move.uci(),
                'probability': float(model_prob),
                'eval_after_move': eval_after_model_move,
                'cp_loss': cp_loss 
            })
        
        return analysis
    
    def run_benchmark(self, num_positions=100, min_moves=10, max_moves=40, show_worst_moves=False, top_k_check=1):
        """Run the benchmark, comparing model(s) to Stockfish over random positions."""
        results = {
            'model_1': {'positions_analyzed': [], 'stats': defaultdict(list)},
            'model_2': {'positions_analyzed': [], 'stats': defaultdict(list)} if self.model_2 else None
        }
        
        for i in tqdm(range(num_positions), desc="Benchmarking Positions"):
            try:
                board = self.generate_random_position(min_moves, max_moves)
                if board.is_game_over():
                    continue
                
                fen = board.fen()
                self.stockfish.set_fen_position(fen)
                stockfish_move_str = self.stockfish.get_best_move()
                if not stockfish_move_str: continue
                stockfish_best_move = chess.Move.from_uci(stockfish_move_str)
                
                model_1_best_pred_move, model_1_top_k = self.get_model_move_and_top_k(self.model_1, board, k=max(top_k_check, 1))
                if not model_1_best_pred_move: continue

                analysis_1 = self.analyze_position(board, model_1_top_k, stockfish_best_move)
                results['model_1']['positions_analyzed'].append(analysis_1)
                
                primary_model_1_move_analysis = analysis_1['model_moves_analysis'][0]
                results['model_1']['stats']['cp_loss'].append(primary_model_1_move_analysis['cp_loss'])
                results['model_1']['stats']['same_as_stockfish_top1'].append(
                    1 if primary_model_1_move_analysis['move'] == stockfish_best_move.uci() else 0
                )
                found_in_top_k_m1 = any(m_analysis['move'] == stockfish_best_move.uci() for m_analysis in analysis_1['model_moves_analysis'][:top_k_check])
                results['model_1']['stats'][f'stockfish_move_in_model_top_{top_k_check}'].append(1 if found_in_top_k_m1 else 0)

                if self.model_2:
                    model_2_best_pred_move, model_2_top_k = self.get_model_move_and_top_k(self.model_2, board, k=max(top_k_check, 1))
                    if not model_2_best_pred_move: continue

                    analysis_2 = self.analyze_position(board, model_2_top_k, stockfish_best_move)
                    results['model_2']['positions_analyzed'].append(analysis_2)
                    
                    primary_model_2_move_analysis = analysis_2['model_moves_analysis'][0]
                    results['model_2']['stats']['cp_loss'].append(primary_model_2_move_analysis['cp_loss'])
                    results['model_2']['stats']['same_as_stockfish_top1'].append(
                        1 if primary_model_2_move_analysis['move'] == stockfish_best_move.uci() else 0
                    )
                    found_in_top_k_m2 = any(m_analysis['move'] == stockfish_best_move.uci() for m_analysis in analysis_2['model_moves_analysis'][:top_k_check])
                    results['model_2']['stats'][f'stockfish_move_in_model_top_{top_k_check}'].append(1 if found_in_top_k_m2 else 0)
                
            except Exception as e:
                print(f"Error analyzing position {i+1}: {e}")
                continue
        
        final_summary_stats = {}
        for model_key, model_data in results.items():
            if model_data is None: continue
                
            num_eval_positions = len(model_data['positions_analyzed'])
            if num_eval_positions == 0:
                final_summary_stats[model_key] = {"error": "No positions successfully evaluated"}
                continue
            
            cp_losses = model_data['stats']['cp_loss']
            model_summary = {
                "positions_evaluated": num_eval_positions,
                "avg_cp_loss_on_top1": np.mean(cp_losses) if cp_losses else 0,
                "median_cp_loss_on_top1": statistics.median(cp_losses) if cp_losses else 0,
                "max_cp_loss_on_top1": max(cp_losses) if cp_losses else 0,
                "min_cp_loss_on_top1": min(cp_losses) if cp_losses else 0,
                "std_dev_cp_loss_on_top1": np.std(cp_losses) if cp_losses else 0,
                "same_move_as_stockfish_rate_top1": (np.mean(model_data['stats']['same_as_stockfish_top1']) * 100) if model_data['stats']['same_as_stockfish_top1'] else 0,
                f"stockfish_move_in_model_top_{top_k_check}_rate": (np.mean(model_data['stats'][f'stockfish_move_in_model_top_{top_k_check}'])*100) if model_data['stats'][f'stockfish_move_in_model_top_{top_k_check}'] else 0,
                "cp_loss_distribution_on_top1": {
                    "<=0 (gain/equal)": len([x for x in cp_losses if x <= 0]),
                    "1-50": len([x for x in cp_losses if 0 < x <= 50]),
                    "51-100": len([x for x in cp_losses if 50 < x <= 100]),
                    "101-200": len([x for x in cp_losses if 100 < x <= 200]),
                    "201-500": len([x for x in cp_losses if 200 < x <= 500]),
                    ">500": len([x for x in cp_losses if x > 500])
                }
            }
            
            if show_worst_moves:
                model_summary["worst_positions_details"] = sorted(
                    model_data['positions_analyzed'], 
                    key=lambda x: x['model_moves_analysis'][0]['cp_loss'] if x['model_moves_analysis'] else -float('inf'), 
                    reverse=True
                )[:5]
            
            final_summary_stats[model_key] = model_summary
        
        return final_summary_stats

def print_model_stats(model_name, stats, top_k_check=1):
    """Helper to print formatted statistics for a model."""
    if "error" in stats:
        print(f"\n{model_name} Results: ERROR - {stats['error']}")
        return

    print(f"\n--- {model_name} Benchmark Results ---")
    print(f"Positions evaluated: {stats['positions_evaluated']}")
    print(f"Avg CP Loss (Top1 vs SF): {stats['avg_cp_loss_on_top1']:.2f}")
    print(f"Median CP Loss (Top1 vs SF): {stats['median_cp_loss_on_top1']:.2f}")
    print(f"Max CP Loss (Top1 vs SF): {stats['max_cp_loss_on_top1']:.2f}")
    print(f"Std Dev CP Loss (Top1 vs SF): {stats['std_dev_cp_loss_on_top1']:.2f}")
    print(f"Same Move as Stockfish (Top1): {stats['same_move_as_stockfish_rate_top1']:.1f}%")
    print(f"Stockfish's Best Move in Model's Top {top_k_check}: {stats[f'stockfish_move_in_model_top_{top_k_check}_rate']:.1f}%")
    
    print("\nCP Loss Distribution (Model's Top1 Move vs Stockfish's Best):")
    for range_name, count in stats['cp_loss_distribution_on_top1'].items():
        percentage = (count / stats['positions_evaluated']) * 100 if stats['positions_evaluated'] > 0 else 0
        print(f"  {range_name}: {count} positions ({percentage:.1f}%)")
    
    if "worst_positions_details" in stats and stats["worst_positions_details"]:
        print(f"\nTop {len(stats['worst_positions_details'])} Positions with Highest CP Loss (Model's Top1 Move):")
        for i, pos_analysis in enumerate(stats["worst_positions_details"]):
            print(f"  {i+1}. FEN: {pos_analysis['fen']}")
            print(f"     Stockfish Best: {pos_analysis['stockfish_move']['move']} (Eval Post-Move: {pos_analysis['stockfish_move']['eval_after_move']})")
            if pos_analysis['model_moves_analysis']:
                model_top_move = pos_analysis['model_moves_analysis'][0]
                print(f"     Model Top1: {model_top_move['move']} (Prob: {model_top_move['probability']:.3f}, Eval Post-Move: {model_top_move['eval_after_move']}, CP Loss: {model_top_move['cp_loss']:.2f})")

if __name__ == "__main__":
    model_1_path = "models/smartyPant-eloFloor2k-ep9.pth"
    model_2_path = "models/selfPlay-v7-ep02-acc01645.pth" 
    # model_2_path = None # Example: Benchmark only one model

    try:
        model_1_dict = torch.load(model_1_path, map_location=torch.device('cpu'))
        print(f"Loaded model 1 from: {model_1_path}")
        
        model_2_dict = None
        if model_2_path and os.path.exists(model_2_path):
            model_2_dict = torch.load(model_2_path, map_location=torch.device('cpu'))
            print(f"Loaded model 2 from: {model_2_path}")
        elif model_2_path:
            print(f"Warning: Model 2 path specified but not found: {model_2_path}")

        from core.encoding import ChessEncoder
        encoder = ChessEncoder()
        
        benchmark_top_k_check = 3 # How many of model's top moves to check if SF best is present
        benchmark_tool = Benchmark(model_1_dict, encoder, model_dict_2=model_2_dict, depth=10)
        results = benchmark_tool.run_benchmark(
            num_positions=100, # Keep low for testing, increase for thorough benchmark
            min_moves=5, 
            max_moves=50, 
            show_worst_moves=True,
            top_k_check=benchmark_top_k_check 
        )
        
        if 'model_1' in results:
            print_model_stats("Model 1", results['model_1'], top_k_check=benchmark_top_k_check)
        if 'model_2' in results and results['model_2']: # Check if model_2 results exist
            print_model_stats("Model 2", results['model_2'], top_k_check=benchmark_top_k_check)

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()