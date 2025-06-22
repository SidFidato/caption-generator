FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y -qq git ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
