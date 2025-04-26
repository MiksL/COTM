# COTM
 Comparison of training methods (self-play and existing games) for chess game agent

## Used packages
pytorch
pytorch-lightning
python-dotenv
numpy
stockfish
tqdm
python-chess
lightning
webdataset
h5py
hdf5plugin

## Env variables to set
PGN_PATH - path to the PGN file from which to read the games from
STOCKFISH_PATH - path to the Stockfish engine file (used for comparing models against stockfish)
GAMES_PATH - directory in which to place pre-processed games (recommended to set as fastest system drive with adequate free space)

## Current model names and their descriptions

### TestBotEG
- Trained on the first 80k games sampled from the lichess database
- Trained for 10 epochs

