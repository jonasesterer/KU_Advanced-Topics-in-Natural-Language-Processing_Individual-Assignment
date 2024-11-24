import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path


# Define the padding token
pad_token = "[PAD]"
start_token = "[SOS]"
end_token = "[EOS]"


# Initialize a set to store unique special tokens
special_tokens = set()
datasetfile = Path(__file__).parent / "datafiles/tasks.txt"
# Read and process the dataset file
# READD FOR SPECIAL TOKENS
#with open(datasetfile, "r") as file:
#    for line in file:
#        # Extract the 'OUT' part using regex
#        matches = re.findall(r'I_\w+', line)
#        if matches:
#            # Split the 'OUT' sequence into tokens
#            # Add tokens starting with 'I_' to the special tokens set
#            special_tokens.update(matches)

# Convert the set to a sorted list
#special_tokens.update({"[EOS]", "[SOS]"})
special_tokens = sorted(special_tokens)
special_tokens.insert(0, pad_token)
special_tokens.insert(1, start_token)
special_tokens.insert(2, end_token)
print(special_tokens)

# Initialize a Byte-Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())

# Define a pre-tokenizer that splits on whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[SOS] $A [EOS]",
    special_tokens=[
        ("[SOS]", 1),
        ("[EOS]", 2),
    ],
)

# Prepare the trainer with the special tokens
trainer = trainers.BpeTrainer(special_tokens=special_tokens)

# Read the entire dataset for training
with open(datasetfile, "r") as file:
    lines = file.readlines()

# Train the tokenizer
tokenizer.train_from_iterator(lines, trainer=trainer)

pad_token_id = tokenizer.token_to_id(pad_token)
start_token_id = tokenizer.token_to_id(start_token)
end_token_id = tokenizer.token_to_id(end_token)
assert pad_token_id == 0, f"Expected {pad_token} to have ID 0, but got {pad_token_id}. ID 0 is {tokenizer.id_to_token(0)}."
assert start_token_id == 1, f"Expected {start_token} to have ID 1, but got {start_token_id}. ID 1 is {tokenizer.id_to_token(1)}."
assert end_token_id == 2, f"Expected {end_token} to have ID 2, but got {end_token_id}. ID 2 is {tokenizer.id_to_token(2)}."

tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)


# Save the tokenizer to a file
tokenizer.save(str( Path(__file__).parent.parent / "custom_tokenizer.json"))
