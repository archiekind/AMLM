from datasets import load_dataset
from tokenizers import Tokenizer
import model
import json
import torch
import matplotlib.pyplot as plt
import random
import os

# hyperparameters
config = json.load(open("./config.json"))
batch_size = config["batch_size"]
block_length = config["block_length"]
training_steps = config["training_steps"]
val_steps = config["val_steps"]
lr = config["learning_rate"]
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# dataset config
print("loading dataset ...", '\n')
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:50%]").shuffle()
dataset_val = load_dataset("cnn_dailymail", "3.0.0", split="validation").shuffle()
tokeniser = Tokenizer.from_file("./tokeniser.json")

def decode(x):
    out = tokeniser.decode(x.view(-1).tolist()).split()
    for i in range(len(out)):
        out[i] = out[i].replace("Ġ", " ")
    print("".join(out))

print("formatting dataset ...", '\n')

def get_input_ids(data):
    out = []
    for example in data:
        encoded = tokeniser.encode(example).ids
        if len(encoded) < block_length + 1:
            padding_length = block_length - len(encoded) + 1
            encoded.extend([tokeniser.token_to_id("[PAD]")] * padding_length)
        out.append(encoded)
    return out

dataset = get_input_ids(dataset["article"])
dataset_val = get_input_ids(dataset_val["article"])

#model init
print("initialising transformer ...", '\n')
transformer = model.Model()
if os.path.exists("./model"):
    transformer.load_state_dict(torch.load("./model"))
transformer.to(device)
optimiser = torch.optim.Adam(transformer.parameters(), lr=lr)

len_dataset = len(dataset)
len_dataset_val = len(dataset_val)


# validation loss estimator
def estimate_val_loss():
    transformer.eval()
    loss_vali = []
    for _ in range(val_steps):
        rndm1 = random.randint(0, (len_dataset_val-batch_size)//batch_size)
        list = dataset_val[batch_size*rndm1 : batch_size*(rndm1+1)]
        for j in range(len(list)):
            rndm2 = random.randint(0, len(list[j]) - block_length - 1)
            list[j] = list[j][rndm2: rndm2 + block_length + 1]
        x = torch.tensor(list)[:,:-1].to(device)
        y = torch.tensor(list)[:,1:].to(device)
        out, loss = transformer(x, y)
        loss_vali.append(loss.item())
    transformer.train()
    return sum(loss_vali)/len(loss_vali)
    


#trainer
train_loss = []
val_loss = []

lossi = []
print("beginning training ...", '\n')
for i in range(training_steps):
    optimiser.zero_grad(set_to_none=True)
    rndm1 = random.randint(0, (len_dataset-batch_size)//batch_size)
    list = dataset[batch_size*rndm1 : batch_size*(rndm1+1)]
    for j in range(len(list)):
        rndm2 = random.randint(0, len(list[j]) - block_length - 1)
        list[j] = list[j][rndm2: rndm2 + block_length + 1]
    x = torch.tensor(list)[:,:-1].to(device)
    y = torch.tensor(list)[:,1:].to(device)
    out, loss = transformer(x, y)
    loss.backward()
    optimiser.step()

    lossi.append(loss.item())

    if i % 500 == 0 and i != 0:
        loss_val = estimate_val_loss()
        loss_train = sum(lossi)/len(lossi)
        lossi = []
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        print(f"step: {i} --- training loss: {loss_train} --- validation loss: {loss_val}")

torch.save(transformer.state_dict(), "./model")


#generate sample
generation = transformer.generate(torch.tensor([32], dtype=torch.long).to(device), 150)
decode(generation)

plt.plot(train_loss)
plt.show()

plt.plot(val_loss)
plt.show()