FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

COPY config/ config/
COPY tests/ tests/

# No data is baked into the image. Mount it at runtime:
#   docker run -v /path/to/mimic:/app/data dementia-undercoding
CMD ["pytest", "-q"]
