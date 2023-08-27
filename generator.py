import model
import torch
from tokenizers import Tokenizer

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokeniser = Tokenizer.from_file("./tokeniser.json")
transformer = model.Model().to(device)
transformer.load_state_dict(torch.load("./model"))

def decode(x):
    out = tokeniser.decode(x.view(-1).tolist()).split()
    for i in range(len(out)):
        out[i] = out[i].replace("Ġ", " ")
    print("".join(out))

prompt = input("Prompt: ")
print('\n')
ids = torch.tensor(tokeniser.encode(prompt).ids, dtype=torch.long).to(device)

response = transformer.generate(ids, 200)

decode(response)