from dataclasses import dataclass


@dataclass
class ProfilerArgs:
    run: bool = False
    trace_folder: str = "profiling"
    mem_warmup: int = 100
    mem_steps: int = 2
    profile_warmup: int = 102
    profile_steps: int = 2


