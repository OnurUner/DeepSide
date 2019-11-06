import torch


def weighted_binary_cross_entropy(input, target, class_weights):
    output = torch.sigmoid(input)
    output = output.clamp(min=0.00001, max=0.9999)

    if class_weights is not None:
        assert len(class_weights) == 2

        loss = torch.mul(class_weights[1].unsqueeze(0), (target * torch.log(output))) + torch.mul(
            class_weights[0].unsqueeze(0), ((1.0 - target) * torch.log(1.0 - output)))
    else:
        loss = target * torch.log(output) + (1.0 - target) * torch.log(1.0 - output)

    return torch.neg(torch.mean(loss))