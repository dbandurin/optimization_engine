FROM python:2.7

ENV ENVCONSUL_VERSION 0.6.2
ENV ENVCONSUL_SHA256 0b12d98b8aaa33fbfd813f1577a0bbfe5b20529fe44cc9a8cbba89df972a881f

RUN set -ex \
    && wget -O envconsul.tgz https://releases.hashicorp.com/envconsul/${ENVCONSUL_VERSION}/envconsul_${ENVCONSUL_VERSION}_linux_amd64.tgz \
    && echo "$ENVCONSUL_SHA256 *envconsul.tgz" | sha256sum -c - \
    && tar -xvzf envconsul.tgz -C /bin/ \
    && rm -rf envconsul.tgz \
    && mkdir /etc/envconsul.d/

RUN apt-get update
RUN apt-get install -y postgresql g++ python-dev musl-dev make cmake

COPY optimization/ /app/optimization
COPY static/ /app/static
COPY ui/ /app/ui
COPY application.py requirements.txt setup.py /app/
COPY etc/ /etc

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/envconsul", "-config", "/etc/envconsul.d/envconsul.hcl"]
CMD python -u application.py