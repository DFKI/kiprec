"""
Topic Recommender class

Topic Recommender class for ranking the course recommendation using the 
semanantic recommendation. The main part of the code is based on the original
internal implementation by Benjamin Paaßen. 

Author: Jakub Kuzilek (jakub.kuzilek@dfki.de), Benjamin Paaßen (benjamin.paaßen@dfki.de)
Version: 0.0.1
Date: 2023-10-11
"""

from .BaseRecommender import BaseRecommender

import numpy as np
from sentence_transformers import SentenceTransformer, util

LANGUAGE_MODEL_ = 'T-Systems-onsite/german-roberta-sentence-transformer-v2'

class TopicRecommender(BaseRecommender):

    def train(self, 
              meta_categories : list[str],
              course_data : list[dict]):
        """ Computes the semantic similarity of each course to the
        given list of meta categories, based on a sentence BERT model.

        Parameters
        ----------
        meta_categories: list
            A list of strings, each specifying a meta category.
        course_data: list[dict]
            A list of dictionaries of all courses. One course should contain 
            the following fields:
            {
                'course_id' : <int>,
                'name' : <str>,
                'tags' : <list[str]>,
                'categories' : <list[str]>
            }

        Returns
        -------
        json_model: dict
            A dictionary with the following fields:
            * meta_categories: a list of meta_category titles from the input.
            * courses: A list of dictionaries of course descriptions, each a dictionary
            with the fields 'course_id', 'name', and 'similarities' (an array
            of similarities to each meta category).

        """

        # prepare sentence BERT model to embed word and phrases
        # we use the T-Systems German sentence transformer
        # because it can deal with both German and English words
        # https://huggingface.co/T-Systems-onsite/german-roberta-sentence-transformer-v2
        model = SentenceTransformer(LANGUAGE_MODEL_, device = 'cpu')

        # embed meta categories
        Xmeta = model.encode(meta_categories)

        # embed course names
        names = []
        for course in course_data:
            names.append(course['name'])
        Xname = model.encode(names)
        # compute cosine similarity based on name embeddings
        Sname = util.cos_sim(Xname, Xmeta).detach().numpy()

        # embed all tags in the data set
        tags = set()
        for course in course_data:
            if 'tags' in course:
                for tag in course['tags']:
                    tags.add(tag)
        tags = list(sorted(tags))
        if len(tags) > 0:
            tag_embed = model.encode(tags)

            # generate an index map for tags
            tag_idxs = {}
            for i in range(len(tags)):
                tag_idxs[tags[i]] = i

            # compute tag embedding for each course
            Xtag = []
            for course in course_data:
                x = np.zeros_like(tag_embed[0, :])
                if 'tags' in course:
                    for tag in course['tags']:
                        i = tag_idxs[tag]
                        x = x + tag_embed[i, :]
                Xtag.append(x)
            Xtag = np.stack(Xtag, 0)
            # compute cosine similarity based on tag embeddings
            Stag = util.cos_sim(Xtag, Xmeta).detach().numpy()
        else:
            Stag = np.zeros((len(course_data), len(meta_categories)))

        # embed all catagories in the data set
        categories = set()
        for course in course_data:
            if 'categories' in course:
                for category in course['categories']:
                    categories.add(category)
        categories = list(sorted(categories))
        if len(categories) > 0:
            cat_embed = model.encode(categories)

            # generate an index map for categories
            cat_idxs = {}
            for i in range(len(categories)):
                cat_idxs[categories[i]] = i

            # compute tag embedding for each course
            Xcat = []
            for course in course_data:
                x = np.zeros_like(cat_embed[0, :])
                if 'categories' in course:
                    for category in course['categories']:
                        i = cat_idxs[category]
                        x = x + cat_embed[i, :]
                Xcat.append(x)
            Xcat = np.stack(Xcat, 0)
            # compute cosine similarity based on tag embeddings
            Scat = util.cos_sim(Xcat, Xmeta).detach().numpy()
        else:
            Scat = np.zeros((len(course_data), len(meta_categories)))

        # compute maximum across all three similarities
        S = np.nanmax(np.stack([Sname, Stag, Scat], 2), 2)

        # start constructing the output object
        json_model = {
            'meta_categories' : meta_categories,
            'courses' : []
        }

        for i in range(len(course_data)):
            course_out = {
                'course_id' : course_data[i]['course_id'],
                'name' : course_data[i]['name'],
                'Sim' : S[i, :].tolist()
            }
            json_model['courses'].append(course_out)

        # return the final similarity model
        self.model = json_model

        return self
    

    def recommend(self,
                  course_data : list[dict],
                  preferences : dict):
        """
        Ranks the courses based on the given student preferences.

        Parameters
        ----------
        course_data: list[dict]
            A list of dictionaries of all courses. One course should contain 
            the following fields:
            {
                'course_id' : <int>,
                'name' : <str>,
                'tags' : <list[str]>,
                'categories' : <list[str]>
            }
        preferences: dict
            A dictionary of preferences, with keys being the meta categories
            and values being the preference value (float).

        Returns
        -------
        ranked_courses: list[dict]
            A list of dictionaries of all courses, sorted by the ranking score.
            Each dictionary contains the following fields:
            {
                'course_id' : <int>,
                'name' : <str>,
                'score' : <float>
            }
        """

        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        
        # normalize preferences
        normalization = sum(preferences.values())
        # if the normalization would be numerically unstable set uniform preferences
        if normalization < 1e-3:
            normalization = 1/len(preferences)

        normalized_preferences = preferences
        [normalized_preferences.update({key: value / normalization}) for key, value in normalized_preferences.items()]

        # reorder normalized preferences
        normalized_preferences =  {key: normalized_preferences[key] for key in self.model['meta_categories']}
        
        # ranked courses
        ranked_courses = []
        for course in course_data:
            crs = {}
            crs['course_id'] = course['course_id']
            crs['name'] = course['name']
            crs['score'] = 0

            if crs['course_id'] not in [d['course_id'] for d in self.model['courses'] if 'course_id' in d]:    
                ranked_courses.append(crs)            
                continue
     
            sim = self.model['courses'][[course['course_id'] for course in self.model['courses']].index(crs['course_id'])]['Sim']
            for i in range(len(sim)):
                crs['score'] += sim[i] * list(normalized_preferences.values())[i]

            print(crs)

            ranked_courses.append(crs)

        # sort the courses by score from highest to lowest
        ranked_courses = sorted(ranked_courses, key=lambda k: k['score'], reverse=True)
            
        return ranked_courses
