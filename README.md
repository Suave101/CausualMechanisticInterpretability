# Causual Mechanistic Interpretability Pipeline

## Quick Start
To get started with the CMI (Causual Mechanistic Interpretability) Pipeline, first you must clone the repository onto your machine locally. You can do this with the following command:
```bash
git clone https://github.com/Suave101/CausualMechanisticInterpretability.git
```
Then you must set your current directory to the CMI repository:
```bash
cd CausualMechanisticInterpretability
```
After that, you must download the required libraries using uv:
```bash
uv sync
```
> Note: If you do not have uv installed, here is the link to do so: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

To run an experiment, you can use the `main.py` wrapper for the project, here is a sample experiment that will run quickly:
```bash
python main.py --model mlp3 --dataset uci_digits --batch_size 32 --top_n_pct 15.0
```

This command will restructure a 3 layer fully-connected mlp based on the uci digits.
