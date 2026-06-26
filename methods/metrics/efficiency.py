def compute_parameters(model):
    """
    Retorna o total de parâmetros do modelo (em milhões).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6
    }
