import logging
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Type, TypeVar
logger = logging.getLogger()
T = TypeVar("T")
def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))
