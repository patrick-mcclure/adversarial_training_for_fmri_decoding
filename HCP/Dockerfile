# pbrain container specification.

# Use "gpu-py3" to build GPU-enabled container and "py3" for non-GPU container.
ARG TF_ENV="gpu-py3"
FROM tensorflow/tensorflow:1.8.0-${TF_ENV}

WORKDIR /opt/pbrain
COPY . .
RUN \
    # Extras do not have to be installed because the only extra is tensorflow,
    # which is installed in the base image.
    pip install --no-cache-dir -e /opt/pbrain \
    && rm -rf ~/.cache/pip/* \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

ENV PATH="$PATH:/opt/pbrain/bin"

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="John Lee <johnleenimh@gmail.com>"
