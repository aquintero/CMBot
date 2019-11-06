FROM python:3.8.0-slim

WORKDIR /tradebot/

COPY ./tradebot/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./tradebot ./

CMD python tradebot.py