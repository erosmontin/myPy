import torch
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b
    