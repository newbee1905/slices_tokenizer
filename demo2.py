import slices_tokenizer as st
from tqdm import tqdm
import random

print("Setting up the demo...")

SOURCE_POOL = [
	"Ga Bi Bi S S S S Cl 0 3 --o 0 5 oo- 0 6 o--",
	"Li Na K 1 2 +++ 1 3 --- 2 4 ooo",
	"Fe O O O 0 1 o-o 0 2 -o- 0 3 --o",
	"Si C N 4 5 +o- 4 6 -o+ 5 7 o-o 5 8 o+o",
	"Au Ag Pt Cu 10 12 --- 11 13 +++ 14 15 ooo",
]

NUM_TRAINING_SAMPLES = 1_000_000

def generate_training_data(pool, num_samples):
	"""A generator that yields random samples from the source pool on-the-fly."""
	for _ in range(num_samples):
		yield random.choice(pool)

print("\n--- Training Tokenizer on 1,000,000 Samples ---")
tokenizer = st.SLICESTokenizer()

training_iterator = generate_training_data(SOURCE_POOL, NUM_TRAINING_SAMPLES)

with tqdm(total=NUM_TRAINING_SAMPLES, desc="Training", unit=" samples") as pbar:
	tokenizer.train(
		iterator=training_iterator,
		vocab_size=500,
		tqdm_instance=pbar
	)

print("Training complete.")

print("\n--- Testing Trained Tokenizer ---")
tokenizer_path = "demo_tokenizer.json"
tokenizer.save(tokenizer_path)

new_tokenizer = st.SLICESTokenizer()
new_tokenizer.load(tokenizer_path)
print(f"Tokenizer loaded from '{tokenizer_path}'")

text = "Ga Si --o ooo +++"
encoding = new_tokenizer.encode(text)

print(f"\nOriginal: '{text}'")
print(f"Tokens:   {encoding.tokens}")
print(f"IDs:	  {encoding.ids}")

decoded = new_tokenizer.decode(encoding.ids)
print(f"Decoded:  '{decoded}'")

print(f"\nVocabulary size: {new_tokenizer.get_vocab_size()}")
print(f"ID for 'ooo': {new_tokenizer.token_to_id('ooo')}")
print(f"ID for 'Ga': {new_tokenizer.token_to_id('Ga')}")
print(f"ID for 'X' (unknown): {new_tokenizer.token_to_id('X')}")
