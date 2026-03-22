from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()


trainer = BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
)

tokenizer.train(files=["ai_corpus_cleaned.txt"], trainer=trainer)

# Save the trained tokenizer
tokenizer.save("ai_subword_tokenizer.json")

sentence = "Agentic AI systems use generative models for autonomous decision making"
encoded = tokenizer.encode(sentence)

print("Tokens:", encoded.tokens)