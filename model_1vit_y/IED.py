import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
import warnings


class JQZFeatEnHancer(nn.Module):		#52.3version
	"""
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales
      2 : Scale aware aggregation of low light features
      3 : Construction of enhanced representation of the input image


      Args :
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

	def __init__(self, hiddim= 32):
		super(JQZFeatEnHancer, self).__init__()

		# int_out_channels = 32
		# out_channels = 24

		# self.ue_conv8 = nn.Conv2d(out_channels, int_out_channels, 3, 1, 1, bias=True)
		# nn.init.constant_(self.ue_conv8.weight,0.0)
		# nn.init.constant_(self.ue_conv8.bias,0.0)

		# print(torch.__version__)
		self.fet_rgb= FET(hiddim=hiddim)
		# self.mean = torch.tensor([0.3958, 0.3729, 0.3715], requires_grad = False).view(1,-1,1,1).cuda()
		# self.std  = torch.tensor([0.1305, 0.1203, 0.1146], requires_grad = False).view(1,-1,1,1).cuda()
		# self.alpha = 0.3

	def forward(self, x):

		g, tail = self.fet_rgb(x)

		return g, tail


# 0703version
class FET(nn.Module):


	def __init__(self, hiddim=32):
		super(FET, self).__init__()

		int_out_channels = hiddim
		self.e_conv1 = nn.Sequential(
			nn.Conv2d(3, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)

		self.e_conv2 = nn.Sequential(
			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)

		self.e_conv3 = nn.Sequential(
			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)

		self.e_conv4 = nn.Sequential(
			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
			nn.ReLU(),
		)

		self.e_conv5 = nn.Sequential(
			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)

		self.e_conv6 = nn.Sequential(
			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)

		self.e_conv7 = nn.Sequential(
			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
			nn.ReLU(),
		)
		self.tail = nn.Sequential(
			nn.Conv2d(int_out_channels, 3, 1, 1, 0, bias=True),
		)
		nn.init.constant_(self.tail[0].weight,0.0)
		nn.init.constant_(self.tail[0].bias,0.0)

		self.gconv = nn.Sequential(
			nn.Conv2d(in_channels=hiddim*2, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True,groups=1),
		)

	def forward(self, x):
		x1 = self.e_conv1(x)
		x2 = self.e_conv2(x1)
		x3 = self.e_conv3(x2)
		x4 = self.e_conv4(x3)
		x5 = self.e_conv5(torch.cat([x3, x4], 1))
		x6 = self.e_conv6(torch.cat([x2, x5], 1))
		x7 = self.e_conv7(torch.cat([x1, x6], 1))

		tail = self.tail(x7)

		xx = torch.cat((x5,x6),dim=1)
		g = self.gconv(xx)

		return  g, tail

# # simple_enet
# class FET(nn.Module):


# 	def __init__(self, hiddim=32):
# 		super(FET, self).__init__()

# 		int_out_channels = hiddim
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(4,int_out_channels,kernel_size=1,padding=0,bias=True),
# 			nn.ReLU(),
# 		)
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels,int_out_channels,kernel_size=5,padding=2,bias=True,groups=4),
# 			nn.ReLU(),
# 		)

# 		self.tail = nn.Sequential(
# 			nn.Conv2d(int_out_channels, 3, 1, 1, 0, bias=True),
# 		)
# 		nn.init.constant_(self.tail[0].weight,0.0)
# 		nn.init.constant_(self.tail[0].bias,0.0)


# 		self.gconv = nn.Sequential(
# 			nn.Conv2d(in_channels=hiddim, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True),
# 			# nn.ReLU(),
# 		)


# 	def forward(self, x):
# 		g = x.mean(dim=1,keepdim=True)
# 		x = torch.cat((x,g),dim=1)


# 		x1 = self.conv1(x)
# 		x2 = self.conv2(x1)
# 		tail = self.tail(x2)

# 		g = self.gconv(x2)
# 		return g, tail
