# Transformer from Scratch for Transliteration

This project implements a Transformer model from scratch using PyTorch. The model is designed for sequence-to-sequence tasks, demonstrated here for transliteration.

## Project Structure

-   `model.py`: Contains the core implementation of the Transformer architecture, including:
    -   `MultiheadAttention` and `CrossMultiheadAttention`
    -   `Encoder` and `Decoder` blocks
    -   `FeedForwardNeuralNetwork`
    -   The main `Transformer` model class that combines the components.
    -   `NoamOpt` for learning rate scheduling.
-   `train.py`: The main script for training the model. It handles data loading, the training loop, evaluation, and checkpointing.
-   `inference.py`: A script to run inference using a pre-trained model checkpoint.
-   `transliteration_data/`: This directory should contain the dataset for training and evaluation.

## Getting Started

### Prerequisites

Make sure you have Python and PyTorch installed. You will also need `numpy` and `wandb` for logging. You can install them using pip:

```bash
pip install torch numpy wandb
```

### Data Preparation

The training and inference scripts rely on helper functions in `transliteration_data/data.py` to handle data preparation. Ensure your dataset is placed in the `transliteration_data` directory.

### Training the Model

To start training, simply run the `train.py` script:

```bash
python train.py
```

You can configure training parameters such as learning rate, batch size, and model dimensions at the top of the `train.py` file. The script is configured to:
-   Log training progress to Weights & Biases (if `wandb_log` is set to `True`).
-   Save the best model checkpoint to the `out/` directory as `ckpt.pt`.

### Running Inference

After training, you can use the `inference.py` script to see the model's predictions on new data.

1.  Make sure you have a trained checkpoint file at `out/ckpt.pt`.
2.  You can modify the sample `sentence` inside the `data_preprocessing` function in `inference.py` to test different inputs.
3.  Run the script:

```bash
python inference.py
```

The script will load the model, process the input sentence, and print the generated transliteration.
