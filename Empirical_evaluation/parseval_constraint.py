"""
    Contains the parseval regularization implementation which was adapted from
    the original paper with its respective available code.
"""

import torch

def parseval_weight_projections(model_temp, scale_param, num_passes=5):
    """
    Implementation of the parseval regularization as proposed in original work
    This implementation was adapted from the TensorFlow one:
        https://github.com/mathialo/parsnet
    """
    # Conv1
    param = model_temp.convs[0].weight.data
    last = param
    for i in range(num_passes):
        temp1 = torch.mm(param.t(), param)
        temp2 = (1 + scale_param) * param - scale_param *  torch.mm(param, temp1)
        last = temp2

    model_temp.convs[0].weight.data = last

    # Conv2
    param = model_temp.convs[1].weight.data
    last = param
    for i in range(num_passes):
        temp1 = torch.mm(param.t(), param)
        temp2 = (1 + scale_param) * param - scale_param *  torch.mm(param, temp1)
        last = temp2

    model_temp.convs[1].weight.data = last
    return model_temp
