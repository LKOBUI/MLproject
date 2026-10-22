FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860
WORKDIR /app

COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/app/
COPY models/ /app/models/
COPY configs/ /app/configs/

EXPOSE 7860
CMD ["python", "-m", "app.interface"]
