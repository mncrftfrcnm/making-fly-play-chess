# making-fly-play-chess

No better way to use fly neurons than to make them play chess!

So, this project uses the fruit fly connectome from [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), and trains it to play chess using reinforcement learning. More precisely, it is a chess agent built from the connectivity of the fruit-fly brain, used as a fixed neural reservoir.

The short explanation is that a chess board gets turned into numbers, those numbers are sent through a network made from fly-neuron connections, and the result is used to score the position. White tries to make the score higher, Black tries to make it lower.

The fly connections stay fixed during training. Only the final part that turns neuron activity into a chess score is learned.

This is a simplified reservoir-computing experiment, not a biological simulation of a living fly brain.

The trained model is already included, so you do not have to train it yourself just to play.

## easiest way to play: google colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mncrftfrcnm/making-fly-play-chess/blob/main/fly_chess_inference.ipynb)

If you do not code and just want to play:

1. Open `fly_chess_inference.ipynb` in Google Colab.
2. At the top, click **Runtime**, then **Run all**.
3. Wait for everything to install and for the model to load.
4. Scroll to the final cell and open the Gradio link.
5. Choose **You vs Fly** or **Fly vs Fly**, then press **Start / New Game**.

In **You vs Fly**, choose White or Black and drag the pieces on the board. The board is enabled only on your turn, and illegal moves snap back automatically.

In **Fly vs Fly**, you can watch two copies of the model play each other. You can change the delay, stop the game after a certain number of moves, or let each fly randomly choose between a few of its best moves.

The interface also shows which moves the fly considered and which neurons were most active after its move.

## what is actually happening?

The board encoder makes 782 input values:

- all 64 squares and 12 possible piece types;
- whose turn it is;
- castling rights;
- en passant;
- the half-move clock.

These values are fed into an 8,192-neuron section of the connectome. Activity moves through the fly connections for a few steps, and 1,024 readout neurons are used to produce a score between `-1` and `+1`.

For every legal move, the program makes the move on a copy of the board and asks the network for a new score. White chooses a high score and Black chooses a low score.

During training, the flies play against themselves. The score is updated from wins, losses, draws, future position values, and a small material reward.

It is more of a strange AI experiment than a serious chess engine, but that is the point.

## running the python files normally

You need Python 3.10 or newer and Git.

Clone the project:

```bash
git clone https://github.com/mncrftfrcnm/making-fly-play-chess.git
cd making-fly-play-chess
```

Make a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the game:

```bash
python fly_chess_inference.py
```

It will print a local address, usually `http://127.0.0.1:7860`. Open that in your browser.

## training it yourself

The trainer needs the original connectome files. Install the extra packages first:

```bash
python -m pip install -r requirements-train.txt
```

Then clone the connectome into the project folder:

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git Drosophila_brain_model
```

Then run:

```bash
python fly_chess_trainer.py
```

The default settings train 3,000 self-play games using 8,192 neurons. This can take a while, so for a quick test you can change:

```python
SELF_PLAY_GAMES = 2
```

Some useful settings near the top of `fly_chess_trainer.py` are:

| Setting | Default | What it changes |
| --- | ---: | --- |
| `SELF_PLAY_GAMES` | `3000` | Number of self-play games |
| `RESERVOIR_NEURONS` | `8192` | Number of fly neurons used |
| `READOUT_NEURONS` | `1024` | Neurons used for the final score |
| `PROPAGATION_STEPS` | `6` | How many times activity moves through the network |
| `MAX_PLIES` | `230` | Maximum half-moves in one game |
| `LEARNING_RATE` | `0.002` | How quickly the value weights change |

If `fly_chess_model.joblib` already exists, training continues from its saved readout weights. The new model is saved to the same file when training finishes.

There is also a Colab-ready training notebook, `fly_chess_trainer.ipynb`. It downloads the connectome data and lets you download the trained model after the run.

## files

| File | What it does |
| --- | --- |
| `fly_chess_inference.py` | Runs the game and the Gradio interface |
| `fly_chess_trainer.py` | Trains the model with self-play |
| `fly_chess_model.joblib` | The included trained model |
| `fly_chess_inference.ipynb` | Google Colab version for playing |
| `fly_chess_trainer.ipynb` | Google Colab version for training |
| `chess_selfplay.gif` | An example fly-vs-fly game |

## how good is it?

I have not measured an Elo for it yet. It only looks one move ahead, so it is not meant to compete with Stockfish.

A good next test would be to play a few hundred games against random moves, a material-only bot, and low Stockfish levels. That would show whether the fly connections are doing anything useful beyond making funny chess games.

## credit

The connectome data comes from [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model).

That project accompanies the paper [*A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing*](https://doi.org/10.1101/2023.05.02.539144).

This repository uses the [Apache License 2.0](LICENSE).
