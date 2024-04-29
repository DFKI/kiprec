"""
Preference-based Recommender class

Preference-based Recommender class for ranking the course recommendation using the 
semanantic recommendation matched with the user preferences. The main part of the 
code is based on the original internal implementation by Benjamin Paaßen. 

Author: Jakub Kuzilek (jakub.kuzilek@dfki.de), Benjamin Paaßen (benjamin.paaßen@dfki.de)
Version: 0.0.3
Date: 2024-02-22
"""

from .BaseRecommender import BaseRecommender

from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer, util

LANGUAGE_MODEL_ = 'aari1995/German_Semantic_STS_V2'

class PreferenceBasedRecommender(BaseRecommender):

    def train(self,
              meta_categories : list[str],
              course_data: list[dict],
              enrollment_data: list[dict] = None):
        """
        Computes the semantic similarity of each course to the
        given list of meta categories plus creates and stores that as a model.

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
        enrollment_data: list[dict] (optional)
            A list of dictionaries of all course enrollments. One enrollment should contain 
            the following fields:
            {
                'user_id' : <int>,
                'enrolments' : <list[dict]>
            }
            where each enrolment should contain the following fields:
            {
                'course_id' : <int>,
                'timestamp' : <str>
            }

        Returns
        -------
        json_model: dict
            A dictionary with the following fields:
            * meta_categories: a list of meta_category titles from the input.
            * features: A list of dictionaries of course features, each a dictionary
                        with the fields 'name' and 'values' (an array of possible values).
            * courses: A list of dictionaries of course descriptions, each a dictionary
                        with the fields 'course_id', 'name', 'tags', 'categories', 'Sim_categories',   
                        'Sim_tags', 'Sim_name', 'Sim', 'features', and 'count' (if enrollment_data is provided).
        """
        
        # iterate over all course_data extract the 'attributes' field and create list of dictionary
        # containing items with two entries - 'name' of subfield from 'attributes' and 
        # 'values' containing all found possible values for that particular subfield plus
        # 'All' as a default value
        features = []
        for course in course_data:
            if 'attributes' in course:
                for key, value in course['attributes'].items():
                    if key not in [feature['name'] for feature in features]:
                        features.append({'name': key, 'values': ['All']})
                    if value not in features[[feature['name'] for feature in features].index(key)]['values']:
                        if isinstance(value, list):
                            for val in value:
                                features[[feature['name'] for feature in features].index(key)]['values'].append(val)
                        else:
                            features[[feature['name'] for feature in features].index(key)]['values'].append(value)     

        model = SentenceTransformer(LANGUAGE_MODEL_, device = 'cpu')   

        # embed meta categories
        Xmeta = model.encode(meta_categories)

        # compute category similarity
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


        # embed course names
        names = []
        for course in course_data:
            names.append(course['name'])
        Xname = model.encode(names)
        # compute cosine similarity based on name embeddings
        Sname = util.cos_sim(Xname, Xmeta).detach().numpy()

        # pre-process all similarities
        Sims = [Scat, Stag, Sname]
        for k in range(len(Sims)):
            lo  = np.nanmin(Sims[k])
            hi  = np.nanmax(Sims[k])
            print('Similarity %d with minimum %g and maximum %g' % (k, lo, hi))
            Sims[k] = (Sims[k] - lo) / (hi - lo)

        # compute maximum across all three similarities
        S = np.nanmax(np.stack(Sims, 2), 2)

        # start constructing the output object
        json_model = {
            'meta_categories' : meta_categories,
            'features' : features,
            'courses' : []
        }

        # iterature over courses and store each course in the
        # output object.
        # In doing so, we also pre-process the course features
        # to be in synch with the filter structure offered on the
        # oncampus website
        for i in range(len(course_data)):
            course = course_data[i]
            out_course = {}
            out_course['name'] = course['name']
            out_course['course_id']   = course['course_id']
            if 'tags' in course:
                out_course['tags'] = course['tags']
            else:
                out_course['tags'] = []
            if 'categories' in course:
                out_course['categories'] = course['categories']
            else:
                out_course['categories'] = []
            out_course['Sim_categories']  = Sims[0][i, :].tolist()
            out_course['Sim_tags']  = Sims[1][i, :].tolist()
            out_course['Sim_name']  = Sims[2][i, :].tolist()
            out_course['Sim']  = S[i, :].tolist()
            out_course['features'] = course['attributes']

            json_model['courses'].append(out_course)

        # if so desired, add course counts
        if enrollment_data is not None:
            # build a histogram of course visits
            counts = Counter()
            for enrollments in enrollment_data:
                for enrolment in enrollments['enrolments']:
                    counts[enrolment['course_id']] += 1

            # transfer the information to our output object
            for i in range(len(course_data)):
                if course_data[i]['course_id'] in counts:
                    count = counts[course_data[i]['course_id']]
                else:
                    count = 0
                json_model['courses'][i]['count'] = count
        else:
            for i in range(len(course_data)):
                json_model['courses'][i]['count'] = 0

        # return the final similarity model
        self.model = json_model
        return self

    def recommend(self, 
                  preferences: dict,
                  active_filters: dict = None):
        """
        Recommend courses based on the preferences and active filters.

        Parameters
        ----------
        preferences: dict
            A dictionary of preferences, with keys being the meta categories
            and values being the preference value (float).
        active_filters: dict
            A dictionary of active filters, with keys being the filter names
            and values being the filter value.

        Returns
        -------
        filters: list[dict]
            A list of dictionaries of active filters, each a dictionary
            with the fields 'name' and 'value'.
        remaining_features: list[str]
            A list of remaining features to filter on.
        filtered_courses: list[dict]
            A list of dictionaries of course descriptions, each a dictionary
            with the fields 'course_id', 'name', 'tags', 'categories', 'Sim_categories',   
            'Sim_tags', 'Sim_name', 'Sim', 'features', and 'count' (if enrollment_data is provided).
        split_index: int   
            The index of the feature to split on in the decision tree.        
        """

        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        
        # filter courses based on the active_filters and preferences
        filters = []
        remaining_features = []

        if active_filters is not None:
            # add active filters
            for feature in self.model['features']:
                if feature['name'] in active_filters and active_filters[feature['name']] in feature['values']:
                    filters.append({'name': feature['name'], 
                                    'value': active_filters[feature['name']]
                                   })
                else:
                    remaining_features.append(feature['name'])
        else: 
            remaining_features = [feature['name'] for feature in self.model['features']]
        
        # filter courses
        filtered_courses = self.__filter_courses__(self.model['courses'],
                                                   filters)
        # rank similarities
        filtered_courses = self.__rank_courses__(filtered_courses,
                                                 preferences)

        # next filter
        split_index = self.__decision_tree_split__(filtered_courses,
                                                   remaining_features)
        return filters, remaining_features, filtered_courses, split_index
    
    def __filter_courses__(self, courses, filters):
        # filter courses based on the filter
        # return the filtered courses
        # for each filter in filters, filter the courses
        for filter in filters:
            courses = [course for course in courses if course['features'][filter['name']] == filter['value']]
        return courses
    
    def __rank_courses__(self, courses, preferences):
        # rank the courses based on the preferences
        # return the ranked courses
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
        for course in courses:
            crs = {}
            crs['course_id'] = course['course_id']
            crs['name'] = course['name']
            crs['features'] = course['features']
            crs['sim'] = 0
            crs['score'] = 0

            if crs['course_id'] not in [d['course_id'] for d in self.model['courses'] if 'course_id' in d]:    
                ranked_courses.append(crs)            
                continue
     
            sim = self.model['courses'][[course['course_id'] for course in self.model['courses']].index(crs['course_id'])]['Sim']
            for i in range(len(sim)):
                crs['sim'] += sim[i] * list(normalized_preferences.values())[i]

            # $score_i = $LAMBDA_ * $sim_i + log($course_list[$i]['count'] + 1);
            _LAMBDA_ = np.log(10) * 10
            crs['score'] = _LAMBDA_ * crs['sim'] + np.log(self.model['courses'][[course['course_id'] for course in self.model['courses']].index(crs['course_id'])]['count'] + 1)

            ranked_courses.append(crs)

        # sort the courses by score from highest to lowest
        ranked_courses = sorted(ranked_courses, key=lambda k: k['score'], reverse=True)

        return ranked_courses
    
    def __decision_tree_split__(self, courses, remaining_features):
        # split the courses based on the decision tree
        # return the split index

        # sum of the scores from all courses
        sum_scores = sum([course['score'] for course in courses])

        if(sum_scores < 1E-3):
            return -1
        
        min_j = -1
        min_loss = float('inf')

        for j in range(len(remaining_features)):
            feature_name = remaining_features[j]

            loss = 0
            non_trivial_choice = False

            # iterate over all possible values of the feature and count the courses with that value
            # store the count for later use
            counts = {}
            for course in courses:
                value = course['features'][feature_name]
                if value not in counts:
                    counts[value] = 0
                counts[value] += 1
            
            # if at least one value has more than one course, and filters out at least one course
            # then we can consider this feature for the split (set non_trivial_choice to True)
            for value in counts:
                if counts[value] > 0 & counts[value] < len(courses):
                    non_trivial_choice = True
                    break
            
            # if non_trivial_choice is False, then we can skip this feature
            if not non_trivial_choice:
                continue

            # at first initiate the minimal course number variable to be equal to number of courses
            # for each course extract score and the value for the feature with name feature_name
            # and count the number of other courses with value of score higher than the current course
            # if the number of courses is less then minimal course number variable then 
            # update the minimal course number variable after the iteration update the loss variable
            min_course_number = len(courses)
            for course in courses:
                score = course['score']
                value = course['features'][feature_name]
                course_number = sum([1 for c in courses if c['features'][feature_name] == value and c['score'] > score])
                if course_number < min_course_number:
                    min_course_number = course_number
                loss += score*course_number

            loss = loss / sum_scores
            if loss < min_loss:
                min_loss = loss
                min_j = j

        return min_j