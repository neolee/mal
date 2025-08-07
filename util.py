from typing import Tuple


def parse_model_str(model: str) -> Tuple[str, str]:
    """Parse the 'provider/model' style model string"""
    assert model
    strs = model.split("/", 1)

    if len(strs) < 2: return strs[0], ""
    return strs[0], strs[1]
