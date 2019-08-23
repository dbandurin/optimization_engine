from setuptools import setup
import os


packages = [
    'optimization',
    'optimization.model',
    'optimization.model.order',
]


def get_version():
    version = None

    with open("etc{sep}VERSION".format(sep=os.sep),
              "r") as f:
        version = f.read().strip()

    return version


setup(
    name='optimization_engine',
    version=get_version(),
    packages=packages,
    setup_requires=[
        "pytest-runner==4.2"
    ],
    extras_require={
        'dev': [
            'mock==2.0.0',
            'more-itertools<6.0.0',
            'pytest==3.6.2',
            'pytest-cov==2.5.1',
            'twine==1.11.0',
            'pylint==1.9.2',
            'more-itertools==5.0.0'
        ]
    },
    install_requires=[
        'bumpversion==0.5.3',
        'numpy==1.14.5',
        'scipy==1.1.0',
        'pandas==0.24.2',
        'boto3==1.7.45',
        'requests==2.19.1',
        'web.py==0.39',
        'psycopg2==2.7.5',
        'freezegun==0.3.10',
        'scikit-learn==0.20',
        'wheel==0.31.1'
    ]
)
