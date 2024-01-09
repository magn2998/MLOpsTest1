# Specify it is from M2 chip and install python
FROM debian:stable-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y python3 python3-pip pipx

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY Test1/ Test1/
COPY data/ data/
COPY Makefile Makefile

RUN mkdir -p reports/figures
RUN mkdir models

# Install Python Requirements
WORKDIR /
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

# Ensure scripts from the virtual environment are used
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 --version
RUN pip install -r requirements.txt

# Entry point
ENTRYPOINT ["make", "train"]
