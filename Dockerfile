FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Movie_Recommender/ Movie_Recommender/

CMD ["gunicorn", "Movie_Recommender.app:app"]
