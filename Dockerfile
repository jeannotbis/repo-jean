FROM python:3.9

ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip

WORKDIR /App

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]