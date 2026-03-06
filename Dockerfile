FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/app/artifacts/mplconfig

COPY pyproject.toml README.md streamlit_app.py ./
COPY src ./src
COPY data ./data
COPY .streamlit ./.streamlit
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN pip install --upgrade pip && pip install ".[database]"
RUN chmod +x docker-entrypoint.sh

EXPOSE 8501

CMD ["./docker-entrypoint.sh"]
