import numpy as np

from src.MLP import MLP
from src.loss import MSE
from src.trainer import Trainer


def make_fake_data(N, mode):
    if mode == 'train':
      x = np.linspace(-3, 3, N)
    if mode == 'test':
      x = (np.random.rand(N) - 0.5) * 6
    y = np.sin(x) + 0.1 * np.random.randn(N)
    x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
    return x, y

# make data
N = 1000
train_x, train_y = make_fake_data(N, 'train')
test_x, test_y = make_fake_data(100, 'test')
train_x = (train_x - train_x.mean()) / train_x.std()

# init model and trainer
config = [1, 10, 10, 1]
model = MLP(config)
loss_fn = MSE()
trainer = Trainer(model, loss_fn, 1e-3)


# =============== begin ================= #
#            gradient descent             #
# =============== begin ================= #
# N_iter = 1000
# num_ = 32
# for iter in range(N_iter):
#         x, y = train_x[0 : num_], train_y[0 : num_]
#         loss = trainer.forward(x, y)
#         print(f'iter: [{iter:>4d}/{N_iter:>4d}], loss: {loss:>4f}')
#         trainer.update_params()
# =============== end =================== #
#            gradient descent             #
# =============== end =================== #



# =============== begin ================= #
#                  SGD                    #
# =============== begin ================= #
# num_epoch = 100
# b_s = 64
# def shuffle_data(x, y, N):
#     idxs = np.arange(N)
#     np.random.shuffle(idxs)
#     return x[idxs], y[idxs]
# for epoch in range(num_epoch):
#     train_x, train_y = shuffle_data(train_x, train_y, N)
#     for batch_idx in range(N // b_s + 1):
#         start_idx = batch_idx * b_s
#         end_idx = start_idx + b_s
#         x, y = train_x[start_idx : end_idx if end_idx < N else N - 1], train_y[start_idx : end_idx if end_idx < N else N - 1]

#         loss = trainer.forward(x, y)
#         if batch_idx % ((N / b_s) // 5) == 0:
#             print(f'epoch [{epoch:>3d}/{num_epoch:>3d}], batch [{batch_idx:>3d}/{(N // b_s):>3d}], loss: {loss:>4f}')
#         trainer.update_params()
# =============== end =================== #
#                 SGD                     #
# =============== end =================== #



# =============== begin ================= #
#               pytorch                   #
# =============== begin ================= #
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

class LRData(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x, self.y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

model = Model()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

num_epoch = 100
b_s = 64
train_dataset = LRData(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=b_s, shuffle=True, drop_last=False)

for epoch in range(num_epoch):
    for batch_idx, (x, y) in enumerate(train_dataloader):
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % ((N / b_s) // 5) == 0:
            print(f'epoch [{epoch:>3d}/{num_epoch:>3d}], batch [{batch_idx:>3d}/{(N // b_s):>3d}], loss: {loss:>4f}')
# =============== end =================== #
#               pytorch                   #
# =============== end =================== #
