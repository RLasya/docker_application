FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install  g++  -y --no-install-recommends && pip install --no-cache-dir -r requirements.txt 

COPY calculator.py /app/

EXPOSE 8501

CMD ["streamlit","run","calculator.py"]

