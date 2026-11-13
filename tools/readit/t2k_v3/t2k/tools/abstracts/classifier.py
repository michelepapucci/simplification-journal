from abc import abstractmethod
from t2k.tools.abstracts.tool import Tool


class Classifier(Tool):

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass
