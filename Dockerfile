FROM python:3.12

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]

# docker build -t python-bentebot .
# docker build --no-cache -t python-bentebot .
# docker run python-bentebot


# docker compose up -d --build