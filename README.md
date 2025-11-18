# AI Chess Bot ♟️

This is my submission to ChessHacks 2025. It's a chess bot powered by a Convolutional Neural Network that evaluates the scores of different game states and feeds that information directly into an optimized move selection algorithm.

## Directory Structure

`/src` is the source code for the bot. It includes `main.py`, and a folder utils which contains all the files used by it.

`/Training` is a copy of the files in `/src` altered to become training code used by `modal_test.py` to train and update `model_weights.pt`. Example training data is added in `/Training/evals` and allows `modal_test.py` to be run without having to add training data (however the model won't be that smart).

`/models` is the folder containing the `model_weights.pt` file that is accessed by the program when running.

`serve.py` is the backend that interacts with the Next.js and bot (`/src/main.py`). It also handles hot reloading of the bot when changes are made to it. This file, after receiving moves from the frontend, will communicate the current board status to the bot as a PGN string, and will send the bot's move back to the frontend.

Unless training, you do not need to run the Python files yourself. The Next.js app includes the `serve.py` file as a subprocess, and will run it for you when you run `npm run dev`.

The backend (as a subprocess) will deploy on port `5058` by default.

## Setup

Start by cloning the repository.

```shell
git clone https://github.com/JummyJoeJackson/AI-Chess-Bot.git
cd AI-Chess-Bot
```

After cloning the repo, add the devtools UI, you can install it with the CLI:

```shell
npx chesshacks install
```

Next, create a Python virtual environment and install the dependencies:

```shell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then, install the dependencies for the Next.js app:

```shell
cd devtools
npm install
```

Afterwards, make a copy of `.env.template` and name it `.env.local` (NOT `.env`). Then fill out the values with the path to your Python environment, and the ports you want to use.

> Copy the `.env.local` file to the `devtools` directory as well.

## Training the Model

Unfortunately, the model in this repository is untrained as the `model_weights.pt` file for the trained model is too large. For the same reason there are only a couple example training data files in `/Training/evals`. If you wish to use more data you must make sure it's formatted in the same way and place it in `/Training/evals` while also changing references to training data in `/Training/modal_test.py`.

First, you will need to go to the [Modal](https://modal.com/) website and create an account, then you can run the following:

```shell
python -m modal setup
```

After that's done, you can navigate to the `/Training` directory and run the training algorithm.

```shell
cd Training
modal run modal_test.py
```

The new model weights and biases will be automatically saved to the `model_weights.pt` file in `/Training/models` (not `/models`), so if you wish to use the trained model make sure to replace the old one in `/models` with the trained model.

## Running the app

Lastly, simply run the Next.js app inside of the devtools folder.

```shell
cd devtools
npm run dev
```

**Visit `http://localhost:3000` in your browser**

## Troubleshooting

First, make sure that you aren't running any `python` commands! These devtools are designed to help you play against your bot and see how its predictions are working. You can see [Setup](#setup) and [Running the app](#running-the-app) above for information on how to run the app. You should be running the Next.js app, not the Python files directly!

If you get an error like this:

```python
Traceback (most recent call last):
  File "/Users/john_doe/code/chess_bot//src/main.py", line 1, in <module>
    from .utils import chess_manager, GameContext
ImportError: attempted relative import with no known parent package
```

you might think that you should remove the period before `utils` and that will fix the issue. But in reality, this will just cause more problems in the future! You aren't supposed to run `main.py ` on your own—it's designed for `serve.py` to run it for you within the subprocess. Removing the period would cause it to break during that step.

### Logs

Once you run the app, you should see logs from both the Next.js app and the Python subprocess, which includes both `serve.py` and `main.py`. `stdout`s and `stderr`s from both Python files will show in your Next.js terminal. They are designed to be fairly verbose by default.

## HMR (Hot Module Reloading)

By default, the Next.js app will automatically reload (dismount and remount the subprocess) when you make changes to the code in `/src` OR press the manual reload button on the frontend. This is called HMR (Hot Module Reloading). This means that you don't need to restart the app every time you make a change to the Python code. You can see how it's happening in real-time in the Next.js terminal.
