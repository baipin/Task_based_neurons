import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import re
from torch.nn import functional as F


class GPUEfficientNeurons(nn.Module):
    def __init__(self, in_features: int, out_features: int, neuron: str, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = neuron

        # 1. 预先解析字符串（只在初始化时执行一次！）
        self.exponents = self._parse_neuron_expr(neuron)
        self.number = len(self.exponents)

        # 2. 用 ModuleList 管理参数，PyTorch 官方标准，完美支持 .to(device) 批量移动
        self.weights = nn.ParameterList([
            Parameter(torch.Tensor(out_features, in_features)) for _ in range(self.number)
        ])

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _parse_neuron_expr(self, neuron_str):
        """在初始化时一次性解析好所有的幂次，转成纯数字"""
        temp = neuron_str.replace(' ', '')
        items = re.split(r'\+|-', temp)
        exponents = []
        for s in items:
            if 'x' in s:
                if '**' in s:
                    # 提取 x**5 里的 5
                    exp = int(s.split('**')[1])
                else:
                    # 只有 x，幂次是 1
                    exp = 1
                exponents.append(exp)
        return exponents  # 比如返回 [5, 3, 2, 1]

    def reset_parameters(self) -> None:
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 3. 纯张量运算，没有任何 exec, eval 和字符串操作！
        # 完美支持 GPU 并行、Cuda Graph、AMP 混合精度和自动求导
        su = 0
        for exponent, weight in zip(self.exponents, self.weights):
            # 依据预先存好的数字幂次进行快速矩阵计算
            if exponent == 1:
                su += F.linear(x, weight, bias=None)
            else:
                su += F.linear(torch.pow(x, exponent), weight, bias=None)

        if self.bias is not None:
            su += self.bias
        return su