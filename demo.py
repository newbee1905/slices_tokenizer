import slices_tokenizer as st

tokenizer = st.SLICESTokenizer()

training_data = [
	"Ga Bi Bi S S S S Cl 0 3 --o 0 5 oo- 0 6 o--",
	"Li Na K 1 2 +++ 1 3 --- 2 4 ooo",
]

tokenizer.train_from_iterator(training_data, vocab_size=1000)

tokenizer.save("my_tokenizer.json")

new_tokenizer = st.SLICESTokenizer()
new_tokenizer.load("my_tokenizer.json")

text = "Ga Bi --o oo- +oo"
encoding = new_tokenizer.encode(text)

print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)

decoded = new_tokenizer.decode(encoding.ids)
print("Decoded:", decoded)

print("Vocab size:", new_tokenizer.get_vocab_size())
print("Bond descriptor '--o' ID:", new_tokenizer.token_to_id("--o"))
