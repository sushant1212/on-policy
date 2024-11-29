import torch

# constants
NOISE_WIDTH = 1.0 / 15
DEBUG = True


def component_log_loss(x, delta, reduction="mean", mask=None):
    """
    Log loss to penalize the number of bits communicated during training
    """
    # print("input shape for loss", x.shape)
    # Loss: log2(|M|z + 1)
    if mask is not None:
        x = x * mask.unsqueeze(-1)

    loss = torch.log2(2 * x.abs() / delta + 1).sum(-1)
        
    # print("loss shape", loss.shape)
    # msg_shape = len(x.shape)
    # if msg_shape == 4:  # message type: key
    #     out = loss[0, 0]  # 1D vector
    # else:  # message type: weighted value
    #     out = loss[0, 0]  # 2D matrix
    if reduction == "none":
        return loss
    return loss.mean(), loss.sum(1).squeeze() 
