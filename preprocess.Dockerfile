FROM apache/beam_python3.12_sdk:2.64.0

ARG WORKDIR=/usr/src/app

ENV PYTHONPATH "${PYTHONPATH}:${WORKDIR}/src"

RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

COPY requirements.in ${WORKDIR}/requirements.txt
RUN pip install -U --no-cache-dir -r requirements.txt

# Copy source code to the container
COPY . ${WORKDIR}/
