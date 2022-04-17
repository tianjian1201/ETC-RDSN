# -*- coding: utf-8 -*-
"""
@project: mutli-test
@author: chenhao
@file: pytorch_tools.py
@ide: PyCharm
@time: 2019/7/11 0011 17:45
@todo: pytorch 实用工具集合
"""
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import numpy as  np


# 参数输出测试
def test():
    model = models.resnet18()
    print("====================")
    for name, parameters in model.named_parameters():
        print(name, ":", parameters)

    print("=====================")
    print(model.layer1[0].conv1.weight.data)
    print(model.layer1[0].conv1.__class__)

    print("=======================")
    input = Variable(torch.randn(20, 16, 50, 100))
    print(input.size())
    print(np.prod(input.size()))


# 模型参数数目统计
def parm_nums():
    model = models.vgg19()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of params: %.2fM" % (total / 1e6))

# 计算模型运算量
def model_flops():

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1 = []
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}
    def simple_hook2(self, input, output):
        list_2["names"] = np.prod(input[0].shape)

    """
    卷积核运算次数的计算
    flop = output_batch * output_weight * output_height * output_channel * params
    因此，主要求出 params 的个数。params 主要包含卷积核+偏置值的数目
    kernel = kernel_width * kernel_height * (in_channel/groups_num)*(2 if multiply_adds else 1)
    """
    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    """
    全连接层计算量
    flop = batch_size * (weight_num + bias_num)
    
    batch_size 与输入向量的维度有关。当输入为[a]时， batch_size = 1；当输入为[[a]]时，batch_size = input.shape[0]
    weight_num：当输入为(1, a)，输出为(1, b)时，weight_num = a * b
    bias_num = 0 或 b
    """
    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    """
    归一化计算量：
    Y = y(x-u)/(o+b)
    可得 计算量为 x 的个数
    """
    list_bn=[]

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[]

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    """
    池化层计算量和 卷积层计算量相似，只是池化层没对特征层分组和相加，也没有偏置值 b
    """
    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
    from model import rdn
    from option import args
    resnet = rdn.RDN(args)
    foo(resnet)
    input = Variable(torch.rand(3, 224, 224).unsqueeze(0), requires_grad=True)
    out = resnet(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

# 输出前向传播的结果
def print_forward():
    model = torchvision.models.resnet18()
    select_layer = model.layer1[0].conv1

    grads = {}
    def save_grad(name):
        def hook(self,input, output):
            grads[name] = input
        return hook

    select_layer.register_forward_hook(save_grad("select_layer"))

    input = Variable(torch.rand(3, 244, 244).unsqueeze(0), requires_grad = True)

    out = model(input)
    print(grads)

# 输出中间值
def print_value():
    grads = {}

    def save_grad(name):
        def hook(grad):
            grads[name] = grad

        return hook

    x = Variable(torch.randn(1, 1), requires_grad=True)
    y = 3 * x
    z = y ** 2

    # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
    y.register_hook(save_grad('y'))
    z.register_hook(save_grad('z'))
    z.backward()
    print('HW')
    print("grads['y']: {}".format(grads['y']))
    print(grads['z'])


# 使用递归输出层数
def print_layer_num():
    resnet = models.resnet18()
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                return 1
        count = 0
        for c in childrens:
            count += foo(c)
        return count

    print(foo(resnet))

# 输出每一层对应权值形状和参数数目
def check_summary():
    def torch_summarize(model, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights."""
        from torch.nn.modules.module import _addindent

        tmpstr = model.__class__.__name__ + ' (\n'
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        return tmpstr

    # Test
    model = models.alexnet()
    print(model)
    print(torch_summarize(model))


# 输出网络结构、参数维度和每层参数数目
def show_summary():
    from collections import OrderedDict
    import pandas as pd

    import torch
    from torch.autograd import Variable
    import torch.nn.functional as F
    from torch import nn

    def get_names_dict(model):
        """
        Recursive walk to get names including path
        """
        names = {}

        def _get_names(module, parent_name=''):
            for key, module in module.named_children():
                name = parent_name + '.' + key if parent_name else key
                names[name] = module
                if isinstance(module, torch.nn.Module):
                    _get_names(module, parent_name=name)

        _get_names(model)
        return names

    def torch_summarize_df(input_size, model, weights=False, input_shape=True, nb_trainable=False):
        """
        Summarizes torch model by showing trainable parameters and weights.

        author: wassname
        url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
        license: MIT

        Modified from:
        - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
        - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/

        Usage:
            import torchvision.models as models
            model = models.alexnet()
            df = torch_summarize_df(input_size=(3, 224,224), model=model)
            print(df)

            #              name class_name        input_shape       output_shape  nb_params
            # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296#(3*11*11+1)*64
            # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
            # ...
        """

        def register_hook(module):
            def hook(module, input, output):
                name = ''
                for key, item in names.items():
                    if item == module:
                        name = key
                # <class 'torch.nn.modules.conv.Conv2d'>
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = module_idx + 1

                summary[m_key] = OrderedDict()
                summary[m_key]['name'] = name
                summary[m_key]['class_name'] = class_name
                if input_shape:
                    summary[m_key][
                        'input_shape'] = (-1,) + tuple(input[0].size())[1:]
                summary[m_key]['output_shape'] = (-1,) + tuple(output.size())[1:]
                if weights:
                    summary[m_key]['weights'] = list(
                        [tuple(p.size()) for p in module.parameters()])

                #             summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])
                if nb_trainable:
                    params_trainable = sum(
                        [torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
                    summary[m_key]['nb_trainable'] = params_trainable
                params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
                summary[m_key]['nb_params'] = params

            if not isinstance(module, nn.Sequential) and \
                    not isinstance(module, nn.ModuleList) and \
                    not (module == model):
                hooks.append(module.register_forward_hook(hook))

        # Names are stored in parent and path+name is unique not the name
        names = get_names_dict(model)

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
        else:
            x = Variable(torch.rand(1, *input_size))

        if next(model.parameters()).is_cuda:
            x = x.cuda()

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        # make dataframe
        df_summary = pd.DataFrame.from_dict(summary, orient='index')

        return df_summary

    # Test on alexnet
    import torchvision.models as models
    model = models.alexnet()
    df = torch_summarize_df(input_size=(3, 224, 224), model=model)
    print(df)


# 显示模型每个特征向量，也可以显示图片。e.g. 特征矩阵为 n*c*h*w 时，在 n**0.5 的矩阵中显示每一个特征向量
def show_save_tensor():
    import torch
    from torchvision import utils
    import torchvision.models as models
    from matplotlib import pyplot as plt

    def vis_tensor(tensor, ch=0, all_kernels=False, nrow=8, padding=2):
        '''
        ch: channel for visualization
        allkernels: all kernels for visualization
        '''
        n, c, h, w = tensor.shape
        if all_kernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        # plt.figure(figsize=(nrow,rows))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))  # CHW HWC
        plt.axis('off')
        plt.ioff()
        plt.show()

    def save_tensor(tensor, filename, ch=0, all_kernels=False, nrow=8, padding=2):
        n, c, h, w = tensor.shape
        if all_kernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
        utils.save_image(tensor, filename, nrow=nrow, normalize=True, padding=padding)

    vgg = models.resnet18(pretrained=True)
    mm = vgg.double()
    filters = mm.modules
    body_model = [i for i in mm.children()][0]
    # layer1 = body_model[0]
    layer1 = body_model
    tensor = layer1.weight.data.clone()
    vis_tensor(tensor)
    save_tensor(tensor, 'test.png')


# 用 Graphviz 生成自动化系统图
"""
Graphviz 是一个可以创建图表的灵活应用程序，可以轻松实现脚本化
安装：
需要先安装 Graphviz 软件；再用 python 安装依赖包
"""
def print_autograd_graph():
    from graphviz import Digraph
    import torch
    from torch.autograd import Variable

    def make_dot(var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph
        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function
        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            # assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    # name = param_map[id(u)] if params is not None else ''
                    # node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')

                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        return dot

    from torchvision import models

    torch.manual_seed(1)
    inputs = torch.randn(1, 3, 224, 224)
    model = models.vgg11_bn(pretrained=False)
    y = model(Variable(inputs))

    g = make_dot(y, params=model.state_dict())
    g.view()


if __name__ == "__main__":
    model_flops()
