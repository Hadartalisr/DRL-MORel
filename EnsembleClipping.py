import torch

def ensemble_clip(gradients, clipping_norm):
    # Compute global norm of all gradients
    global_norms = [torch.norm(torch.cat([g.flatten() for g in grad_set])) for grad_set in gradients]

    # Clip gradients
    clipped_gradients = []
    for grad_set, global_norm in zip(gradients, global_norms):
        if global_norm > clipping_norm:
            scaling_factor = clipping_norm / global_norm
            clipped_gradients.append([g * scaling_factor for g in grad_set])
        else:
            clipped_gradients.append(grad_set)

    return clipped_gradients
