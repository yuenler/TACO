FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/coco

ENV PYTHONPATH=/app

CMD ["bash"]
