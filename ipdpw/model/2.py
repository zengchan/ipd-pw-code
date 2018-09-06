import torch
import numpy as np
from torch.autograd import Variable
a = torch.Tensor([[2,3],  [4,8], [7,9]])
# print('a is:{}'.format(a))
# print('a size is {}'.format(a.size()))
#
# b = torch.LongTensor([[2,3],  [4,8], [7,9]])
# print('b is :{}'.format(b))
#
# c = torch.zeros((3,2))
# print('zero tensor:{}'.format(c))
#
# d = torch.randn((3,2))
# print('normal randon is :{}'.format(d))
#
# a[0,1] = 100
# print('changed a is:{}'.format(a))

# numpy_b = b.numpy()
# print('conver to numpy is :\n {}'.format(numpy_b))

# e = np.array([[2,3],[4,5]])
# torch_e = torch.from_numpy(e)
# print('from numpy to torch.Tensor is {}'.format(torch_e))
#
# f_torch_e = torch_e.float()
# print('change data type to float tensor:{}'.format(f_torch_e))
#
# if torch.cuda.is_available():
#     a_cuda = a.cuda()
#     print(a_cuda)

# x = Variable(torch.Tensor([1]), requires_grad=True)
# w = Variable(torch.Tensor([2]), requires_grad=True)
# b = Variable(torch.Tensor([3]), requires_grad=True)
#
# y = w*x+b
# y.backward()
# print(x.grad)
# print(w.grad)
# print(b.grad)

# x = torch.randn(3)
# x = Variable(x,requires_grad=True)
#
# y = x*2
# print(y)
#
# y.backward(torch.FloatTensor([1,0.1,0.01]))
# print(x.grad)

# class myDataset(Dataset):
#     def _init_(self,csv_file,txt_file,root_dir,other_file):
#         self.csv_data = pd.read_csv(csv_file)
#         with open(txt_file,'r') as f:
#             data_list = f.rea;ines()
#         self.txt_data = data_list
#         self.root_dir = root_dir
#     def _len_(self):
#         return len(self.csv_data)
#     def _getitem_(self, idx):
#         data = (self.csv_data[idx], self.txt_data[idx])
#         return data
#
class net_name(nn.Module):
    def _init_(self, other_arguments):
        super(net_name, self)._init_()
        self.conv1 = nn.Conv2d(in_channels, out_channels ,kernel_size)
    def forward(self, x):
        x = self.conv1(x)
        return x