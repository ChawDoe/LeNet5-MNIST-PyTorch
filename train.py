from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist


if __name__ == '__main__':

	train_data = mnist.MNIST(root='./train', train=True, download=True)
	test_data = mnist.MNIST(root='./test', train=False, download=True)

	random_X = np.random.randn(1, 1, 32, 32)
	model = Model()
	random_y = model.forward(torch.FloatTensor(random_X))
