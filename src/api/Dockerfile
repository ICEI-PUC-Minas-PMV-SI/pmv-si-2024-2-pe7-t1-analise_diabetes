FROM python:3.8-slim AS treinamento

WORKDIR /build

COPY build/dataset-full.csv .

COPY build/requirements.build.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY build/build.*.py .

RUN python3 build.naive.py
RUN python3 build.tree.py
RUN python3 build.random.py

FROM python:3.8-slim AS api

WORKDIR /app

COPY requirements.prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=treinamento /build/naive_model.pkl .
COPY --from=treinamento /build/random_model.pkl .
COPY --from=treinamento /build/tree_model.pkl .

COPY app.py .

ENV ENV=production
EXPOSE 5000
CMD [ "python3", "app.py" ]