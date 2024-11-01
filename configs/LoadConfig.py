import json
from typing import Any, Dict


class DotDict(dict):
    """
    一个字典类，支持使用点“.”操作符访问值。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                self[key] = [DotDict(item) for item in value]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_json_config(file_path: str) -> DotDict:
    """
    加载 JSON 配置文件，并将其转换为 DotDict 对象。

    Args:
        file_path: JSON 配置文件的路径。

    Returns:
        DotDict: 包含配置信息的 DotDict 对象。
    """
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return DotDict(config_data)

def config_to_json(args,file_path):
    with open(file_path, 'wt') as f:
        json.dump(args, f, indent=4)
# if __name__ == "__main__":
#     config = load_json_config('config.json')
#     print(config.dataset)  # 输出 "HAR"
#     print(config.batch_size)  # 输出 32