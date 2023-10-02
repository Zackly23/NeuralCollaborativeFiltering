import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MovieLensDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        userID = self.data[index][0]
        itemID = self.data[index][1]
        target = self.target[index]

        return (userID, itemID), target

class NFCModel(nn.Module):
    def __init__(self, n_user, n_item, n_embed, n_hidden):
        super(NFCModel, self).__init__()
        self.user_embed = nn.Embedding(num_embeddings=n_user, embedding_dim=n_embed)
        self.item_embed = nn.Embedding(num_embeddings=n_item, embedding_dim=n_embed)

        self.fc = nn.Sequential(
            nn.Linear(in_features=n_embed*2, out_features=n_hidden*4),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=n_hidden*4, out_features=n_hidden*4),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=n_hidden*4, out_features=n_embed),

        )

        self.layer = nn.Linear(n_embed*2,1)
        self.Flatten = nn.Flatten()

    def forward(self, user, item):
        p_embed = self.user_embed(user)
        q_embed = self.item_embed(item)

        # print('p_embed : ',p_embed.shape)
        # print('q_embed : ',q_embed.shape)
        gmf = p_embed*q_embed
        # print('gmf : ',gmf.shape)
        mlp_concat = torch.concat((p_embed, q_embed), dim=-1)
        # print('mlp_concat : ',mlp_concat.shape)
        mlp = self.fc(mlp_concat)

        mf_concat = torch.concat((gmf, mlp), dim=-1)
        logit = self.layer(mf_concat)

        return logit

data = pd.read_csv('Movielens.csv')

X = data.drop(columns=['rating']).to_numpy()
y = data.rating.to_numpy()

unique_user = len(np.unique(X[:,0])) + 1
unique_movie = len(np.unique(X[:,1])) + 1
print(unique_movie, unique_user)

X = torch.tensor(X, dtype=torch.int64)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train[0,:])

trainSet = MovieLensDataset(X_train, y_train)
testSet = MovieLensDataset(X_test, y_test)

trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = DataLoader(testSet, batch_size=4, shuffle=True)

model = NFCModel(n_user=unique_user, n_item=unique_movie, n_embed=128, n_hidden=32)

criterion = nn.MSELoss()
optimizer = optim.AdamW(params=model.parameters(), lr=0.001)
EPOCHS = 50

trainLoss, testLoss = [], []
for epoch in range(EPOCHS):
    trainloss = 0
    iterTrain = 0
    for (userid, movieid), target in trainLoader:
        logit = model(userid.view(-1,1), movieid.view(-1,1)).squeeze()
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iterTrain += 1
        trainloss += loss.item()

    trainLoss.append(trainloss/iterTrain)
    iterTest = 0
    testloss = 0
    with torch.no_grad():
        for (userid, movieid), target in testLoader:
            logit = model(userid.view(-1, 1), movieid.view(-1, 1)).squeeze()
            loss = criterion(logit, target)
            testloss += loss.item()
            iterTest += 1

        # print(testloss)
        testLoss.append(testloss/iterTest)

    print(f"Epoch {epoch} | trainLoss {np.mean(trainLoss):.3f} | testLoss {np.mean(testLoss):.3f}")


# dataIter = iter(testLoader)
# with torch.no_grad():
#     for _ in range(5):
#         (userid, movieid), target = next(dataIter)
#         logit = model(userid.view(-1, 1), movieid.view(-1, 1)).squeeze()
#         print(logit)
#         print(target)


plt.plot(trainLoss, label='Training')
plt.plot(testLoss, label='Testing')
plt.legend()
plt.show()

