FROM python:3.9

ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN pip install --upgrade pip

WORKDIR /App

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]
