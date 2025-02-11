from services.config.config_model import ParametrizationAgent


class Config:

    def __init__(self, params_data: dict):
        self.parametrization = ParametrizationAgent(**params_data)