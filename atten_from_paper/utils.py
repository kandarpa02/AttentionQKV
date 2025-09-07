def count_params(params):
    """Recursively count total number of parameters in a nested dict."""
    total = 0
    for v in params.values():
        if isinstance(v, dict):
            total += count_params(v)
        else:
            total += v.size
    return total
