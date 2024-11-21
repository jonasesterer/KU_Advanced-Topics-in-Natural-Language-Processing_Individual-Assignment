import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize a set to store unique special tokens
special_tokens = set()
datasetfile = "/home/klaus/Desktop/DTU Semester 9/KU-NLP/data/datafiles/SCAN/tasks.txt"
# Read and process the dataset file
with open(datasetfile, "r") as file:
    for line in file:
        # Extract the 'OUT' part using regex
        match = re.match(r"OUT:\s*(.*)", line)
        if match:
            # Split the 'OUT' sequence into tokens
            tokens = match.group(1).split()
            # Add tokens starting with 'I_' to the special tokens set
            special_tokens.update(token for token in tokens if token.startswith("I_"))

# Convert the set to a sorted list
special_tokens.update(("<EOS>", "<SOS>"))
special_tokens = sorted(special_tokens)

# Initialize a Byte-Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())

# Define a pre-tokenizer that splits on whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Prepare the trainer with the special tokens
trainer = trainers.BpeTrainer(special_tokens=special_tokens)

# Read the entire dataset for training
with open(datasetfile, "r") as file:
    lines = file.readlines()

# Train the tokenizer
tokenizer.train_from_iterator(lines, trainer=trainer)

# Save the tokenizer to a file
tokenizer.save("custom_tokenizer.json")

print(
    f"Tokenizer trained with {len(special_tokens)} special tokens and saved as 'custom_tokenizer.json'."
)
