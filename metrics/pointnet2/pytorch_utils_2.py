import torch.nn as nn
import torch
from typing import List, Tuple
from Common.utilities import Self_Attn

class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )
        #self.add_module("local_attn",Self_Attn(in_dim=args[-1]))


class DenseSharedMLP(nn.Module):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()
        c_in = args[0]
        self.dense_modules = nn.ModuleList()
        for i in range(len(args) - 1):
            c_out = args[i + 1]
            self.dense_modules.append(
                Conv2d(
                    c_in,
                    c_out,
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm,
                    name='dense_layer{}'.format(i)
                )
            )
            c_in = c_in + c_out
        #self.add_module("local_attn",Self_Attn(in_dim=args[-1]))

    def forward(self, x):

        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](x)
            x = torch.cat([x, y], 1)

        return y



class MultiDenseMLP_1D(nn.Module):

    def __init__(
            self,
            mlps: List[int], #for output
            mlps2: List[int], # for inputs
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()
        c_in = mlps2[0]
        assert len(mlps) == len(mlps2)
        self.dense_modules = nn.ModuleList()
        for i in range(len(mlps)):
            c_out = mlps[i]
            self.dense_modules.append(
                Conv1d(
                    c_in,
                    c_out,
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm,
                    name='dense_layer{}'.format(i)
                )
            )
            if i < len(mlps)-1:
                c_in = c_in + c_out + mlps2[i+1]
        #self.add_module("local_attn",Self_Attn(in_dim=args[-1]))

    def forward(self, x: List):
        pc = x[0]
        size = len(self.dense_modules)
        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](pc)
            if i < size - 1:
                pc = torch.cat([pc, y, x[i+1]], 1)

        return y


class MultiDenseMLP_2D(nn.Module):

    def __init__(
            self,
            mlps: List[int], #for output
            mlps2: List[int], # for inputs
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()
        c_in = mlps2[0]
        assert len(mlps) == len(mlps2)
        self.dense_modules = nn.ModuleList()
        for i in range(len(mlps)):
            c_out = mlps[i]
            self.dense_modules.append(
                Conv2d(
                    c_in,
                    c_out,
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm,
                    name='dense_layer{}'.format(i)
                )
            )
            if i < len(mlps)-1:
                c_in = c_in + c_out + mlps2[i+1]
        #self.add_module("local_attn",Self_Attn(in_dim=args[-1]))

    def forward(self, x: List):
        pc = x[0]
        size = len(self.dense_modules)
        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](pc)
            if i < size - 1:
                pc = torch.cat([pc, y, x[i+1]], 1)

        return y


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

