# 1.导入包
import torch
import torch.nn.functional as F      # 包含激活函数
import torch.utils.data as Data      # 批训练模块
import matplotlib.pyplot as plt      # 绘图工具
# 2.hyper parameters 超参数
LR = 0.01          # 学习率
BATCH_SIZE = 32    # 一批训练32个数据
EPOCH = 12         # 所有数据迭代训练12次

# 3.数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)

# 4.搭建和定义网络
class Net(torch.nn.Module):
    # 设置神经网络属性，定义各层的信息
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  # 隐藏层 hideen layer
        self.predict = torch.nn.Linear(20, 1)  # 输出层 output layer
    # 前向传递过程
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
# 定义4个网络，分别用不同的优化器优化
net_SGD =Net()
net_Momentum =Net()
net_RMSprop =Net()
net_Adam =Net()
nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]  #将4个网络放在一个list当中

# 5.优化器
# 4种经典的优化器：SGD、Momentum、RMSprop、Adam
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
# 用于回归的误差计算公式
loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]  # 保存loss

# 6.训练
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            prediction = net(x)
            loss = loss_func(prediction, y)
            opt.zero_grad()               # 为下次训练清空梯度
            loss.backward()               # 误差反向传播，计算梯度
            opt.step()                        # 更新梯度
            l_his.append(loss.data)  #保存loss

# 7.打印可视化
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
# print(losses_his)
# print(len(losses_his))
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.savefig("c.png")
plt.show()
