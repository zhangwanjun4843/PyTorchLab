def get_json_value(outputs, key: str, default=None):
    if not isinstance(outputs, dict):
        return default
    return outputs.get(key, default)
