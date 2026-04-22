FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lekton_ambiyans_api.py .

EXPOSE 8000

CMD ["uvicorn", "lekton_ambiyans_api.py:app", "--host", "0.0.0.0", "--port", "8000"]
