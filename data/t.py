from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "/home/klaus/Desktop/DTU Semester 9/KU-NLP/custom_tokenizer.json"
)
out = tokenizer.encode(
    "I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_LEFT <EOS>"
)
