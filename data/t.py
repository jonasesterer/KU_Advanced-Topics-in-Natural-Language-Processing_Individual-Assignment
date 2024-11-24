from tokenizers import Tokenizer
from pathlib import Path

path = str(Path(__file__).parent.parent / "custom_tokenizer.json")
print(path)
tokenizer: Tokenizer = Tokenizer.from_file(
path
)


#from tokenizers.processors import TemplateProcessing
#tokenizer.post_processor = TemplateProcessing(
#    single="[SOS] $A [EOS]",
#    special_tokens=[
#        ("[SOS]", tokenizer.token_to_id("[SOS]")),
#        ("[EOS]", tokenizer.token_to_id("[EOS]")),
#    ],
#)


out = tokenizer.encode(
    "I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_LEFT"
)
print(out.ids)
print(tokenizer.decode(out.ids))
