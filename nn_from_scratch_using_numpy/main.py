import numpy as np

from MLP import MLP
from src.loss.mse import MSE
from src.optim.sgd import SGD
from utils.ploter import Ploter

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


# =============== begin ================= #
#             custom implement            #
# =============== begin ================= #

# init model and trainer
config = [1, 10, 10, 1]
model = MLP(config)
loss_fn = MSE()
optimizer = SGD(0.1, 0.9)

# sort test data for ploter
idx_ = np.argsort(test_x, axis=0)
test_x = test_x[idx_]
test_y = test_y[idx_]
ploter = Ploter(test_x, test_y)


N_iter = 1000
b_s = 64

for iter in range(N_iter):
        # get minibatch for SGD
        sample_idx = np.random.randint(0, len(train_x), (b_s, ))
        x, y = train_x[sample_idx], train_y[sample_idx]

        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        l_back = loss_fn.backward()
        model.backward(l_back)

        optimizer.step(model)
        ploter.update_pred(model.forward(test_x))
        print(f'iter: [{iter:>4d}/{N_iter:>4d}], loss: {loss:>4f}')

ploter.show(show=True)
# =============== end =================== #
#             custom implement            #
# =============== end =================== #



# =============== begin ================= #
#               pytorch                   #
# =============== begin ================= #
# import torch
# from torch import nn
# from torch import optim
# from torch.utils.data import Dataset, DataLoader

# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(1, 10),
#             nn.ReLU(),
#             nn.Linear(10, 10),
#             nn.ReLU(),
#             nn.Linear(10, 1)
#         )

#     def forward(self, x):
#         return self.layers(x)

# class LRData(Dataset):
#     def __init__(self, x, y) -> None:
#         super().__init__()
#         self.x, self.y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return len(self.x)

# model = Model()
# optimizer = optim.SGD(model.parameters(), lr=1e-2)
# criterion = nn.MSELoss()

# num_epoch = 100
# b_s = 64
# train_dataset = LRData(train_x, train_y)
# train_dataloader = DataLoader(train_dataset, batch_size=b_s, shuffle=True, drop_last=False)


# test_x = torch.from_numpy(test_x.astype(np.float32))
# test_y = torch.from_numpy(test_y.astype(np.float32))
# idx_ = test_x.argsort(dim=0)
# test_x = test_x[idx_]
# test_y = test_y[idx_]

# ploter = Ploter(test_x, test_y)

# for epoch in range(num_epoch):
#     for batch_idx, (x, y) in enumerate(train_dataloader):
#         pred = model(x)
#         loss = criterion(pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch_idx % ((N / b_s) // 5) == 0:
#             print(f'epoch [{epoch:>3d}/{num_epoch:>3d}], batch [{batch_idx:>3d}/{(N // b_s):>3d}], loss: {loss:>4f}')
        
#         with torch.no_grad():
#             pred = model(test_x)
#             ploter.update_pred(np.array(pred))

# ploter.show()


# =============== end =================== #
#               pytorch                   #
# =============== end =================== #
