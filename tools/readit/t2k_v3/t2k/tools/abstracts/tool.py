from abc import ABC, abstractmethod


class Tool(ABC):

    @abstractmethod
    def bootstrap(self, **kwargs):
        pass

    @abstractmethod
    def analyze(self, X, **kwargs) -> dict:
        pass
