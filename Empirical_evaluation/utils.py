"""
Utils script containing utils function to be used
"""

import torch
import torch.nn.functional as F

from parseval_constraint import parseval_weight_projections

def train(model_local, optimizer_local, data_local, adj_normalized):
    model_local.train()
    optimizer_local.zero_grad()
    out = model_local(data_local.x, adj_normalized)
    loss = F.cross_entropy(out[data_local.train_mask], data_local.y[data_local.train_mask])
    loss.backward()
    optimizer_local.step()
    return model_local, float(loss)

def parseval_train(model_local, optimizer_local, data_local, adj_normalized, retraction_par = 0.001):

    model_local.train()
    optimizer_local.zero_grad()
    out = model_local(data_local.x, adj_normalized)
    loss = F.cross_entropy(out[data_local.train_mask], data_local.y[data_local.train_mask])
    loss.backward()
    optimizer_local.step()
    model_local = parseval_weight_projections(model_local, retraction_par)
    return model_local, float(loss)


@torch.no_grad()
def test(model_local, data_local, adj_normalized):
    model_local.eval()
    pred = out = model_local(data_local.x, adj_normalized).argmax(dim=-1)

    accs = []
    for mask in [data_local.train_mask, data_local.val_mask, data_local.test_mask]:
        accs.append(int((pred[mask] == data_local.y[mask]).sum()) / int(mask.sum()))
    return accs

def compute_acc_perturbation(model_local, data_local, adj_local, x_pertubed_local):

    model_local.eval()
    out_1 = model_local(data_local.x, adj_local)
    pred_1 = out_1.argmax(dim=-1)
    acc_1 = int((pred_1[data_local.test_mask] == data_local.y[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

    out_2 = model_local(x_pertubed_local, adj_local)
    pred_2 = out_2.argmax(dim=-1)
    acc_2 = int((pred_2[data_local.test_mask] == data_local.y[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

    return acc_1, acc_2, out_1, out_2


if __name__ == "__main__":
    pass
