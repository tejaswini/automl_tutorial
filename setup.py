from setuptools import setup, find_packages

setup(name='automl_tutorial',
      version='1.0',
      description='Code accompanying automl tutorial',
      author='Tejaswini Pedapati',
      author_email='tejaswini.pedapati@gmail.com',
      url='',
      packages=find_packages(),
      python_requires='>=3.8',
      install_requires=[
          'ray[tune]==2.3.0',
         'pandas==1.5.2',
          'bayesian-optimization==1.2.0',
          'numpy==1.20.3',
          'scikit-learn==1.2.0',
          'hpbandster==0.7.4',
          'ConfigSpace==0.4.18',
          'hyperopt==0.2.5',
          'matplotlib'
      ])
