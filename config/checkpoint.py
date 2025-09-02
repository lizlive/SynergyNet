from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class SaveEvery:
    every: int = 1
    keep: int = 1


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    eval: SaveEvery = field(default_factory=SaveEvery)
    path: Optional[str] = None
    init_ckpt_path: Optional[str] = None
    continue_training_from_init: bool = False



