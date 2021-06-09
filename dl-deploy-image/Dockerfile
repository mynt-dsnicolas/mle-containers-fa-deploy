FROM fastdotai/fastai:2021-02-11
USER root
RUN apt-get update && apt-get -y upgrade
RUN useradd --create-home deploy
WORKDIR /home/deploy
USER deploy
RUN mkdir /home/deploy/requirements
WORKDIR /home/deploy/requirements
COPY requirements.txt .
USER root
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
USER deploy
COPY opt/ /home/deploy/opt/
WORKDIR /home/deploy/opt/dl/code
USER root
ENTRYPOINT ["gunicorn"]
CMD ["--workers=3", "--bind", "0.0.0.0:8080", "deploy:app"]