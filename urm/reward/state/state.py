from urm.reward.state.interface.env_interface import EnvInterface
from typing import Optional


class State:
    def __init__(self, **kwargs):
        self.env_condition: Optional[EnvInterface] = kwargs.get('env', None)
