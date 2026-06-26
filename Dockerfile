FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

CMD ["python", "-m", "methods.laqda.cli.train"]
