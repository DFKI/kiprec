"""
Recommender abstract class

Base class from which all other recommenders inherits 
basic structure.

Author: Jakub Kuzilek (jakub.kuzilek@dfki.de)
Version: 0.0.1
Date: 2023-10-11
"""

from abc import ABC, abstractmethod
import json

class BaseRecommender(ABC):

    # model is a dictionary containing the model parameters
    model = None

    # method for training the model - the input data varies based on the model
    @abstractmethod
    def train(self):
        pass

    # method for recommendation - the output varies based on the model
    @abstractmethod
    def recommend(self):
        pass

    def store_model(self, file):
        """ Store the model to a file.

        Parameters
        ----------
        file: str
            A string specifying the file path.
        """
        if self.model is not None:
            with open(file,'w') as f:
                json.dump(str(self.model), f)
        else:
            raise ValueError('Model is empty! You need to train the model first.') 