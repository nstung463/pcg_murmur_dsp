FROM python:3.11-slim

WORKDIR /physionet
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY README.md ./README.md
ENV PYTHONPATH=/physionet/src
CMD ["python", "-m", "pcg_dsp.cli", "--help"]
