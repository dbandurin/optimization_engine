import os

# loads the configurations for the databases from environment variables.
DATABASE = {
    'kevin': {
        "user": os.environ.get('KEVIN_USER'),
        "password": os.environ.get('KEVIN_PASSWORD'),
        "database": os.environ.get('KEVIN_DB'),
        "host": os.environ.get('KEVIN_HOST'),
        "port": os.environ.get('KEVIN_PORT')
    },
    'metrics': {
        "user": os.environ.get('METRICS_USER'),
        "password": os.environ.get('METRICS_PASSWORD'),
        "database": os.environ.get('METRICS_DB'),
        "host": os.environ.get('METRICS_HOST'),
        "port": os.environ.get('METRICS_PORT')
    }
}

# loads the ASW configurations for S3 from environment variables.
S3 = {
    'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
}
