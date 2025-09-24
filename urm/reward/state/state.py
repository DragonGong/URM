from urm.reward.state.interface.env_interface import EnvInterface
from typing import Optional


class State:
    def __init__(self, env: EnvInterface, **kwargs):
        self.env_condition: EnvInterface = env
