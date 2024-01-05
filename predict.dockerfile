# Specify it is from M2 chip and install python
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY Test1/ Test1/
COPY data/ data/
COPY Makefile Makefile

RUN mkdir -p reports/figures
RUN mkdir models

# Install Python Requirements
WORKDIR /
RUN pip install . --no-cache-dir #(1)

# Entry point
ENTRYPOINT ["make", "predict"]



