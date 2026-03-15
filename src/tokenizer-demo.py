import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import Block

# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Using a Tokenizer
# why should we use our own tokenizer. Because your dataset can be unique to you
# Domain specific data or specific data
#  Eg: Indian specific tokens eg: samosa -> your own tokenizer will understand this
#  Pretrained tokenizer will not understand this
import sentencepiece as spm

import os
corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input=corpus_path,
    model_prefix="tokenizer",
    vocab_size=40,
    model_type="bpe"
)

# tokenizer.model (trained model)
# Split the words into tokens
# tokenizer.vocab (tokens)

sp = spm.SentencePieceProcessor()
# Load the tokenizer model
sp.load("tokenizer.model")

# Encode the text into tokens
ids = sp.encode(text, out_type=int) # list
data = torch.tensor(ids, dtype=torch.long)
print(data)
vocab_size = sp.get_piece_size()

# Using pretrained tokenizer



# context window is how mych data can be stored at a time. i.e. block size
# LLM will see prev 6 words to generate next 6 words.
block_size = 6
# your model will have 32 value for each word i.e. related words
embedding_dim = 32
# number of self attention heads
n_heads = 2
# number of transformer blocks
n_layers = 2
# learning rate
lr = 1e-3

# number of times model will see the data
epochs = 1500

def get_batch(batch_size = 16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # [17, 33, 3, 37, 38, 8, 24, 13, 26, 1, 21, 8, 0, 32, 26, 30, 8, 24, 34, 27, 36, 37, 18, 8, 22, 26, 9, 11, 10, 8, 24, 19, 26, 39, 16, 8, 5, 25, 35, 41, 20, 2, 13, 8, 15, 26, 0, 7, 6, 8, 14, 40, 4, 20, 12, 8, 31, 28, 24, 29, 23, 8]
    # 12 (token12, token13, token14, token15, token16, token17)
    # for every ix, we get block_size tokens
    x = torch.stack([data[i:i+block_size] for i in ix])
    # for every ix, we get block_size tokens starting from i+1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #x = [token1, token2, token3, token4, token5, token6]
    #y = [token2, token3, token4, token5, token6, token7]
    # LLM will predict token2 for token1, token3 for token2, and so on.
    return x, y

class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # token embedding - mapping each token to a vector of 32 floating point values
        # EG: 2 - hello - [32 floating point values] = [0.66, 0.03, ...]
        # 3 - friends - [32 floating point values]
        # 4 - how - [32 floating point values]
        # 5 - are - [32 floating point values]
        # 6 - you - [32 floating point values]
        # 7 - <END> - [32 floating point values]
        # what are these floating point values? 
        # they are random values that are learned during training
        # they are the weights of the neural network
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # (42, 32)
        # matrix of vocab_size i.e. no of words * 32 floating point values. 
        # this will be random initially
        # As you train your mode, the 32 values becomes close to the word
        
        # positional embedding - order of word, position of each word
        # EG: 0 - [32 floating point values]
        # 1 - [32 floating point values]
        # 2 - [32 floating point values]
        # 3 - [32 floating point values]
        # 4 - [32 floating point values]
        # 5 - [32 floating point values]
        # what are these floating point values? 
        # they are random values that are learned during training
        # they are the weights of the neural network
        self.positional_embedding = nn.Embedding(block_size, embedding_dim) # (6, 32)
        # matrix of block_size i.e. context window * 32 floating point values. 
        
        self.block = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        # Layer Normalization
        self.ln_f = nn.LayerNorm(embedding_dim)
        # Linear layer to map the output of the transformer blocks to the vocabulary size
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # B - batch size, T - block size(context length) 16, 6
        # idx and targets are both (B, T)
        tok_emb = self.embedding(idx) # (B, T, C) 16, 6, 32
        pos_emb = self.positional_embedding(torch.arange(T, device=idx.device)) # (T, C) 6, 32
        x = tok_emb + pos_emb # (B, T, C) 16, 6, 32 why to add? 
        # because AI doesn't understand order of words, so we add positional embedding to the token embedding
        # so that AI can understand the order of words
       
        # What it does: Passes the data through the main "brain" of the model (a Transformer block).
        # In simple terms: This step tries to understand the context of the sentence. It uses Self-Attention to figure out how each word relates to all the other words (e.g., realizing "bank" means a river bank and not a financial bank based on the surrounding words).
        # Then, it uses a Feed Forward network to process and deepen this understanding.
        # (B, T, C) 16, 6, 32 means your data is shaped as 16 independent sentences (Batch), 6 words per sentence (Time), and 32 features describing each word (Channels).
        x = self.block(x)

        # What it does: Applies "Layer Normalization" to the data.
        # In simple terms: As data passes through the complex math in step 1, 
        # the numbers can get wildly huge or super tiny, which confuses the model.
        # Layer Normalization acts like a stabilizer—it scales all the numbers back to a standard, healthy range. 
        # This makes the model learn much faster and more reliably.
        x = self.ln_f(x) # (B, T, C) 16, 6, 32 Layer Normalization

        # What it does: Uses a final linear layer (the "language modeling head") to make predictions.
        # In simple terms: Now that the model has deeply processed the context of the sentence, 
        # this final step translates that understanding into actual predictions for what the next word should be.
        # "Logits" is just a fancy machine learning term for "raw scores." 
        # The model is outputting a massive list of scores—one for every possible word in its vocabulary—indicating how likely each word is to come next.
        logits = self.lm_head(x) # (B, T, vocab_size) 16, 6, 42 Linear layer to map the output of the transformer blocks to the vocabulary size
        
        # Loss Calculation Explanation:
        # if targets is None: We are just generating text (not training), so we don't have true "target" words to compare against.
        if targets is None:
            # So, we don't calculate an error/loss.
            loss = None
        else:
            # 1. Unpack the shape of the predictions.
            # B: Batch (number of sentences, batch_size), T: Time (words per sentence, block_size), C: Channels (vocab size).
            B, T, C = logits.shape
            # 2. PyTorch's cross entropy function expects a 2D list of predictions, 
            # so we flatten the "Batch" and "Time" dimensions together into a single list of words.
            # We calculate the "loss" (how wrong the model is) by comparing its 
            # predictions to the actual correct next words. Lower loss = better predictions.
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # What Softmax does:
            # F.softmax is a mathematical function that takes raw scores (logits) 
            # and converts them into percentages (probabilities) that add up to 100% (or 1.0).
            # What dim=-1 means:
            # We apply softmax across the *last dimension* of the data (the list of vocabulary words)
            # to figure out the probability of each individual word being the next word.
            probs = F.softmax(logits, dim=-1)
            # Fetch the word with highest probability
            idx_next = torch.multinomial(probs, num_samples=1)
            # add the word with highest prob to input sequence. Eg hello -> hello friends -> hello friends how -> hello friends how are ....
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = TinyLLM()

# What torch.optim.AdamW does:
# It is an optimization algorithm that updates the model's parameters to minimize the loss.
# It is a variant of the Adam optimization algorithm that includes weight decay.
# What model.parameters() does:
# It returns an iterator over the model's parameters.
# What lr does:
# It is the learning rate, which is the step size of the optimization algorithm.
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# What torch.optim.AdamW does:
# It is an optimization algorithm that updates the model's parameters to minimize the loss.
# It is a variant of the Adam optimization algorithm that includes weight decay.
# What model.parameters() does:
# It returns an iterator over the model's parameters.
# What lr does:
# It is the learning rate, which is the step size of the optimization algorithm.


# train model for epochs
for step in range(epochs):
    # sample a batch of data
    xb, yb = get_batch()
    # forward pass
    logits, loss = model(xb, yb)
    # backward pass
    optimizer.zero_grad()
    # update the weights
    loss.backward()
    optimizer.step()
    # print loss every 100 steps
    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss.item()}")

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# generate text
context = torch.tensor([sp.encode("hello")], dtype=torch.long)
out = model.generate(context, max_new_tokens=20)

print("\nGenerated text:\n")

generated_ids = out[0].tolist()
print(sp.decode(generated_ids))