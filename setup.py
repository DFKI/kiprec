from setuptools import setup

setup(
    name='kiprec',
    version='1.0.0.dev1',    
    description='''kiprec is a Python package for course recommendations in the context of vocational
education. It contains multiple recommenders employing various Machine Learning
approaches.''',
    url='',
    author='Benjamin Paaßen, Jakub Kuzilek',
    author_email='benjamin.paassen@dfki.de, jakub.kuzilek@dfki.de',
    license='Apache',
    packages=['kiprec'],
    install_requires=['numpy','sentence_transformers'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: POSIX :: Linux', 
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Private :: Do Not Upload'
    ],

    keywords=''
)