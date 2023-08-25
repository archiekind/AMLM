from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import model
import torch
import matplotlib.pyplot as plt
import random

# hyperparameters
batch_size = 32
block_length = 40
training_steps = 15000
lr = 1e-3
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# dataset

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10%]").shuffle()
#tokeniser = Tokenizer(BPE(unk_token="[UNK]"))
#tokeniser.pre_tokenizer = ByteLevel()
#trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], vocab_size=150)
#tokeniser.train(files=["./data.txt"], trainer=trainer)
#tokeniser.save("./tokeniser.json")
tokeniser = Tokenizer.from_file("./tokeniser.json")

def decode(x):
    out = tokeniser.decode(x).split()
    for i in range(len(out)):
        out[i] = out[i].replace("Ġ", " ")
    print("".join(out))

enc = tokeniser.encode("Hello, baby!")
decode(enc.ids)

input_ids = []
for article in dataset["article"]:
    encoded = tokeniser.encode(article)

    if len(encoded.ids) < block_length:
        padding_length = block_length - len(encoded.ids)
        encoded.ids.extend([block_length.token_to_id("[PAD]")] * padding_length)

    input_ids.append(encoded.ids)


transformer = model.Model().to(device)
optimiser = torch.optim.Adam(transformer.parameters(), lr=lr)

dataset = input_ids
len_dataset = len(input_ids) - 20

#trainer
losses = []

for i in range(training_steps):
    optimiser.zero_grad(set_to_none=True)
    rndm = random.randint(0, len_dataset//batch_size)
    list = dataset[batch_size*rndm:batch_size*(rndm+1)]
    for j in range(len(list)):
        list[j] = list[j][:block_length+1]
    x = torch.tensor(list)[:,:-1].to(device)
    y = torch.tensor(list)[:,1:].to(device)
    out, loss = transformer(x, y)
    loss.backward()
    optimiser.step()

    if (i-1) % 10 == 0:
        losses.append(loss.item())
        print(loss.item())

#generate sample
generation = transformer.generate(torch.tensor([32], dtype=torch.long).to(device)).view(-1).tolist()
decode(generation)

plt.plot(losses)
plt.show()