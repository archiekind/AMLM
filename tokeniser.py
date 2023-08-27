from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokeniser = Tokenizer(BPE(unk_token="[UNK]"))
tokeniser.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], vocab_size=2000)
tokeniser.train(files=["./data.txt"], trainer=trainer)
tokeniser.save("./tokeniser.json")