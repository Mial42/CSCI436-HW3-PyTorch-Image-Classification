"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

## MLPModel()
class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, dropout=0.6, input_size=28,fc_hidden_1=32,fc_hidden_2=64,fc_hidden_3=64, k_size=4,stride=1):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------
		
		## define CNN layers below
		self.conv = nn.Sequential( 	
									nn.Conv2d(1, fc_hidden_1,kernel_size=k_size,stride=stride,padding=1), #Only 1 inpjut channel
									nn.BatchNorm2d(fc_hidden_1), # BatchNorm2d layer
									nn.ReLU(),# activation fun,
									nn.Dropout(dropout),# dropout,

									nn.Conv2d(fc_hidden_1,fc_hidden_2,kernel_size=k_size,stride=stride,padding=1),
									nn.BatchNorm2d(fc_hidden_2), # BatchNorm2d layer
									nn.ReLU(),# activation fun,
									nn.Dropout(dropout),# dropout,
									 # BatchNorm2d layer

									nn.Conv2d(fc_hidden_2,fc_hidden_3,kernel_size=k_size,stride=stride,padding=1),
									nn.BatchNorm2d(fc_hidden_3), # BatchNorm2d layer
									nn.ReLU(),# activation fun,
									nn.Dropout(dropout),# dropout,
									## continue like above,
									## **define pooling (bonus)**,
									nn.MaxPool2d(kernel_size = 2, stride = 2)


								)

		##------------------------------------------------
		## write code to define fully connected layer below
		##------------------------------------------------
		
		in_size = fc_hidden_3 * 12 * 12 
		out_size = 10 #Because we're predicting 10 classes
		self.fc = nn.Sequential(
							nn.Linear(in_size, fc_hidden_1),
							nn.BatchNorm1d(fc_hidden_1), # BatchNorm1d layer
							nn.Linear(fc_hidden_1,out_size),
							nn.BatchNorm1d(out_size) 
							)

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------
		x_out = self.conv(x)

		## write flatten tensor code below (it is done)
		x = torch.flatten(x_out,1) # x_out is output of last layer
		

		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc(x)   # predict y
		result = F.softmax(result, dim=1)
		
		return result
        
		
		
	
		