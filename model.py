from torch.nn import Module
from torch import nn


class Model(Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.pool2 = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(400, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		y = self.conv1(x)
		y = self.pool1(y)
		y = self.conv2(y)
		y = self.pool2(y)
		y = y.view(y.shape[0], -1)
		y = self.fc1(y)
		y = self.fc2(y)
		y = self.fc3(y)

		return y
