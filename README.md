# runnnig the app
Simply execute the following command to run the research-ui
```
python applicaiton.py
```

# configuration
This application expects the following environment variables to be set.
```
KEVIN_USER
KEVIN_PASSWORD
KEVIN_DB
KEVIN_HOST
KEVIN_PORT

OPTIMIZATION_USER
OPTIMIZATION_PASSWORD
OPTIMIZATION_DB
OPTIMIZATION_HOST
OPTIMIZATION_PORT

RECOMMENDATION_USER
RECOMMENDATION_PASSWORD
RECOMMENDATION_DB
RECOMMENDATION_HOST
RECOMMENDATION_PORT

UTOPSA_USER
UTOPSA_PASSWORD
UTOPSA_DB
UTOPSA_HOST
UTOPSA_PORT

ADWORDS_USERNAME
ADWORDS_PASSWORD

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```


# tests
Simply execute the following command to run tests
```
pytest tests/
```
or
```
python setup.py test
```

# docker
To build the container
```
docker build -t <name>:<version> .
```
To run the container
```
docker run -d -e CONFIG_NAME:config_value .... -e OTHER_CONFIG_NAME:other_config_value <name>:<version>
```