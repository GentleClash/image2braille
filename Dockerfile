FROM python:3.13.1-slim
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_HOME /root
WORKDIR $APP_HOME
COPY . $APP_HOME/

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
