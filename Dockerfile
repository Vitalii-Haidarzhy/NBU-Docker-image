FROM python:3.8

WORKDIR /app

COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock

RUN pip install pipenv uvicorn cuda-python
RUN pipenv sync

COPY ./python_scripts/main.py /app/
COPY ./python_scripts/trained_vgg16.ml /app/

CMD ["pipenv",  "run",  "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
