FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app

RUN pip install .

COPY /src /app/src

RUN pip install .

COPY .streamlit /app/.streamlit

CMD ["streamlit", "run", "src/painting_tutor/app/app.py", "--server.port", "8080"]
