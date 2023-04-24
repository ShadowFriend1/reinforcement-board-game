import os
from distutils.core import setup

# Optional project description in README.md:
current_directory = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Console
    console=['src/main.py'],

    # Project name:
    name='reinforcement-board-game',

    # Project version number:
    version='v1.0',

    # List a license for the project, eg. MIT License
    license='GNU General Public License v2.0',

    # Short description of your library:
    description='Train and play against reinforcement learning models',

    # Long description of your library:
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Your name:
    author='Jack Negus',

    # Your email address:
    author_email='jack.negus@city.ac.uk',

    # Link to your GitHub repository or website:
    url='https://github.com/ShadowFriend1/reinforcement-board-game',

    # Download Link from where the project can be downloaded from:
    download_url='https://github.com/ShadowFriend1/reinforcement-board-game/archive/refs/heads/main.zip',

    # List project dependencies:
    install_requires=['tensorflow~=2.12.0',
                      'tf_agents',
                      'scipy~=1.10.1',
                      'numpy', 'matplotlib~=3.7.1',
                      'setuptools~=65.6.3',
                      'gym~=0.23.0',
                      'scikit-learn~=1.2.2',
                      'pyglet~=1.5.0',
                      'cloudpickle',
                      'pandas~=1.5.3',
                      'seaborn~=0.12.2',
                      'ipython~=8.12.0',
                      'pysimplegui']
)
