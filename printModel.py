from model import DenseNet
from torchsummary import summary

model = DenseNet()
summary(model, (3, 224, 224))
