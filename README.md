# TinyLLM

A miniature, educational PyTorch implementation of a GPT-style Transformer language model. 

This project is built from scratch to demonstrate the core components of modern Large Language Models (LLMs) in a simple, understandable way. It trains on a tiny hard-coded "corpus" of text and learns to predict the next word, allowing it to generate simple sentences.

## Features

- **Custom Transformer Blocks**: Implements Self-Attention and Feed-Forward Neural Networks.
- **Embeddings**: Includes both Token Embeddings (mapping words to mathematical vectors) and Positional Embeddings (giving the model an understanding of word order).
- **Layer Normalization**: Stabilizes the learning process.
- **Text Generation**: Implements a simple text generation loop using Softmax probabilities and multinomial sampling.

## Project Structure

- `src/transformer_block.py`: Contains the architecture for the core `Block` (Transformer Block) which forms the "brain" of the model.
- `src/demo.py`: Contains the `TinyLLM` model definition, the training loop, and the text generation script. 

## How It Works 

The code is heavily commented to explain complex machine learning concepts in simple terms, answering questions like:
- Why do we add Positional Embeddings to Token Embeddings?
- What exactly do Self-Attention and Feed Forward blocks do?
- How does Cross-Entropy Loss flatten and grade predictions?
- How does the Softmax function convert raw math scores into readable percentage probabilities?

## Usage

To train the model and generate a sample sentence, run:

```bash
python src/demo.py
```

### Expected Output
The model will train for 1500 epochs on its tiny dataset. Once finished, it will use a starting context word (like "hello") to probabilistically generate the next 15 words.

Example output:
```
hello friends how are you <END> the tea is very hot <END> my name is Aarohi
```
