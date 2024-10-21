# building with AWS initially
# FROM public.ecr.aws/lambda/python:3.12
FROM python:3.12

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache


COPY app.py params.json config.py .env crag_agent.py ./
COPY poetry.lock pyproject.toml ./
COPY services ./services
COPY crag ./crag

# DISABLE CACHE
RUN poetry install --without test --no-root && rm -rf $POETRY_CACHE_DIR

# ENABLE VIRTUALENV
ENV VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:$PATH"

# OPEN PORT
EXPOSE 8080
# Command can be overwritten by providing a different command in the template directly.

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["fastapi", "run", "app.py", "--port", "80"]
# ENTRYPOINT ["poetry", "run", "python", "-m", "app"]
