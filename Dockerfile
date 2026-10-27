FROM python:3.12

WORKDIR /code
RUN apt-get update && apt-get install -y --no-install-recommends libturbojpeg0
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .

ENV DISABLE_CONTRACTS=1
ENV MPLBACKEND=Agg

RUN python3 -m pip install --no-deps .

# run it once to see everything OK
CMD ["make", "test"]
