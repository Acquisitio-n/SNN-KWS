# model.py  【70 K + GroupNorm + Dropout】
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional

class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2.0, surrogate_function=neuron.surrogate.ATan())
        #替代梯度函数，解决脉冲不可导问题，反向传播用。

def conv1d(c1, c2, k, s=1, p=0, d=1):
    return nn.Sequential(
        nn.Conv1d(c1, c2, k, stride=s, padding=p, dilation=d, bias=False),
        # nn.BatchNorm1d(c2)                 # 原 BatchNorm
        nn.GroupNorm(8, c2)                  # ← 抗小 batch，零参数量
    )

class GLSC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.local = conv1d(c1, c2, 3, p=1)
        self.global_ = conv1d(c1, c2, 3, p=2, d=2)
        self.sn = PLIF()

    def forward(self, x):
        y = self.local(x) + self.global_(x)
        return self.sn(y)

class BottleneckPLIF(nn.Module):
    def __init__(self, c):
        super().__init__()
        c4 = c // 4                        #先降维（c → c/4），再升维（c/4 → c），减少计算量。
        self.f1 = nn.Sequential(conv1d(c, c4, 1), PLIF())
        self.f3 = nn.Sequential(conv1d(c4, c4, 3, p=1), PLIF())
        # 残差支路加 Dropout（零参数量正则）
        self.f1_out = nn.Sequential(
            conv1d(c4, c, 1),
            PLIF(),
            nn.Dropout(0.1)      # ← 仅残差通路，主通路不变
        )

    def forward(self, x):
        return x + self.f1_out(self.f3(self.f1(x)))

class KWS_SNN(nn.Module):
    def __init__(self, n_class=12, T=8):
        super().__init__()
        self.T = T
        # 70 K 通道配置
        self.stage = nn.Sequential(
            GLSC(1, 32),
            GLSC(32, 64),
            nn.AvgPool1d(2),
            GLSC(64, 64),
            nn.AvgPool1d(2),
            GLSC(64, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        self.b_neck = nn.Sequential(
            BottleneckPLIF(64),
            BottleneckPLIF(64)
        )
        self.fc = nn.Linear(64, n_class)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)          # (T,B,1,L)
        x = functional.seq_to_ann_forward(x, self.stage)    # (T,B,64,1)
        x = functional.seq_to_ann_forward(x, self.b_neck)   #将时间维度当作 batch 维度，复用 ANN 模块
        x = x.mean(0).squeeze(-1)                           # (B,64)
        return self.fc(x)

# 快速验证
if __name__ == "__main__":
    net = KWS_SNN()
    from torchsummary import summary
    summary(net, (1, 16000), device="cpu")