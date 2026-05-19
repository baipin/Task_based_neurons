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

        # Pre-parse the string
        self.exponents = self._parse_neuron_expr(neuron)
        self.number = len(self.exponents)

        #  ModuleList used to admin paras
        self.weights = nn.ParameterList([
            Parameter(torch.Tensor(out_features, in_features)) for _ in range(self.number)
        ])

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _parse_neuron_expr(self, neuron_str):
        temp = neuron_str.replace(' ', '')
        items = re.split(r'\+|-', temp)
        exponents = []
        for s in items:
            if 'x' in s:
                if '**' in s:
                    # e.g. get 5 from x**5
                    exp = int(s.split('**')[1])
                else:
                    # e.g. get 1 from x
                    exp = 1
                exponents.append(exp)
        return exponents  # e.g. return [5, 3, 2, 1]

    def reset_parameters(self) -> None:
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Pure tensor operations
        su = 0
        for exponent, weight in zip(self.exponents, self.weights):
            if exponent == 1:
                su += F.linear(x, weight, bias=None)
            else:
                su += F.linear(torch.pow(x, exponent), weight, bias=None)

        if self.bias is not None:
            su += self.bias
        return su