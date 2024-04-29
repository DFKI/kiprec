# test the TopicRecommender
import kiprec.TopicRecommender as tr

meta_categories = ['IT', 'Business', 'Health']

course_data = []

course_data.append({'course_id': 1,
                    'name': 'Python for Data Science',
                    'categories': ['data_science_programming'],
                    'tags': ['Python', 'Data Science'],
                    'attributes': {'duration': '4 weeks',
                                   'badges': 'yes',
                                   'certificate': 'A',
                                   'price': 'free',
                                   'type': 'mooc_course'
                                   }
                    })
course_data.append({'course_id': 2,
                    'name': 'Python for Business',
                    'categories': ['business_programming'],
                    'tags': ['Python'],
                    'attributes': {'duration': '2 months',
                                   'badges': 'no',
                                   'certificate': 'C',
                                   'price': '100 EUR',
                                   'type': 'blended_course'
                                   }
                      })
course_data.append({'course_id': 3,
                    'name': 'Python for Health',
                    'categories': ['it_in_medicine'],
                    'tags': ['Python'],
                    'attributes': {'duration': '2 months',
                                   'badges': 'no',
                                   'certificate': 'A',
                                   'price': '50 EUR',
                                   'type': 'blended_course'
                                   }
                    })
course_data.append({'course_id': 4,
                    'name': 'Python for Business and Health',
                    'categories': ['business_programming', 'it_in_medicine'],
                    'tags': ['Python'],
                    'attributes': {'duration': '4 weeks',
                                   'badges': 'no',
                                   'certificate': 'A',
                                   'price': 'free',
                                   'type': 'blended_course'
                                   }
                    })

enrollment_data = []

enrollment_data.append({'student_id': 1,
                        'enrolments': [{'course_id': 1, 
                                        'date': '2021-10-01'},
                                       {'course_id': 2, 
                                        'date': '2020-01-01'}
                                       ]
                        })

enrollment_data.append({'student_id': 2,
                        'enrolments': [{'course_id': 1, 
                                        'date': '2018-03-07'},
                                       {'course_id': 3, 
                                        'date': '2021-09-30'}
                                       ]
                        })

enrollment_data.append({'student_id': 3,
                        'enrolments': [{'course_id': 2, 
                                        'date': '2021-10-01'},
                                       {'course_id': 3, 
                                        'date': '2021-10-01'}
                                       ]
                        })

# create category similarity object
print("TopicRecommender test:")

recommender = tr().train(meta_categories, course_data)

print(recommender.model)

print(recommender.recommend(course_data, {'IT': 0.2, 'Business': 0.6, 'Health': 0.3}))

#--------------------------------------------------------------------------------------------
# test the PreferenceBasedRecommender
print("PreferenceBasedRecommender test:")

import kiprec.PreferenceBasedRecommender as pbr

recommender = pbr().train(meta_categories, course_data, enrollment_data) 

print(recommender.model)

filters, features, courses, split_index = recommender.recommend({'IT': 0.2, 'Business': 0.6, 'Health': 0.3}, 
                                          {'duration': '4 weeks','price': 'free'})

[print(f) for f in filters]

print(features)

[print(c) for c in courses]

print(split_index)