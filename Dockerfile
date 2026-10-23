FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/mitocensus
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[biology]'

ENTRYPOINT ["mitocensus"]
CMD ["--help"]
