"""
kiprec

kiprec is a Python library for course recommendations in the context of vocational
education. It contains multiple recommenders employing various Machine Learning
approaches.
"""

__version__ = "'1.0.0.dev1'"
__author__ = 'Benjamin Paaßen, Jakub Kuzilek'
__credits__ = 'DFKI'

from . import TopicRecommender
from .TopicRecommender import TopicRecommender
from .PreferenceBasedRecommender import PreferenceBasedRecommender