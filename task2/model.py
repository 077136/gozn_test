import torch.nn as nn


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.encoder=nn.Sequential(
					nn.Conv2d(1,3,(1,5)),
					nn.SELU(True),
					nn.BatchNorm2d(3),
					nn.Conv2d(3,3,5),
					nn.SELU(True),
					nn.Conv2d(3,3,5),
					nn.ReLU(True),
					  )
		
		self.decoder=nn.Sequential(
					nn.ConvTranspose2d(3,3,5),
					nn.SELU(True),
					nn.ConvTranspose2d(3,3,5),
					nn.BatchNorm2d(3),
					nn.SELU(True),
					nn.ConvTranspose2d(3,1,(1,5)),
					  )
	
 
	def forward(self,x):
		x = x.unsqueeze(1)
		x=self.encoder(x)
		x=self.decoder(x)
		x = x.squeeze(1)
		return x