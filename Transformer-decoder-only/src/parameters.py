import torch

from model import Model


def calculate_parameters(model, path):
    model = Model()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    
    # calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params

if __name__=='__main__':
    model = Model()
    total_params = calculate_parameters(model, './model/model.pth')
    print(f"Total parameters: {total_params}")