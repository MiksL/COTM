# COTM
 Comparison of training methods (self-play and existing games) for chess game agent

## Setting up locally
* Run ```pip install -r requirements.txt``` from the root directory
* Nvidia GPU highly recommended (not tested on CPU inference)

## Env variables to set
1. PGN_PATH - path to the PGN file from which to read the games from
2. STOCKFISH_PATH - path to the Stockfish engine file (used for comparing models against stockfish)
3. GAMES_PATH - directory in which to place pre-processed games

## Current best models
### Self-play
#### v7
### Existing games
#### smartyPant
* Trained on ~2 million lichess games from December 2024 with an elo floor of 2000
* Consists of 20 residual blocks
* Trained for 9 epochs
* Performs similar to Stockfish 17 AVX2 when Elo limited to 2750

<div align="center">

| MCTS simulations | Stockfish Elo | Win | Draw | Loss |
| :--------------: | :-------------: | :---: | :----: | :----: |
| 800              | 2000            | 4     | 0      | 0      |
| 1000             | 2000            | 4     | 0      | 0      |
| 800              | 2250            | 4     | 0      | 0      |
| 1000             | 2250            | 3     | 1      | 0      |
| 800              | 2500            | 3     | 1      | 0      |
| 1000             | 2500            | 4     | 0      | 0      |

</div>

<div align="center">

| MCTS simulations | Win | Draw | Loss |
| :--------------: | :---: | :----: | :----: |
| 800              | 10    | 3      | 7      |
| 1000             | 8     | 5      | 7      |
| 1200             | 5     | 6      | 9      |
| 1400             | 11    | 6      | 3      |
| 1600             | 5     | 9      | 6      |

</div>


