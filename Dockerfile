FROM public.ecr.aws/lambda/python:3.13

COPY hair_classifier_empty.onnx.data .
COPY hair_classifier_empty.onnx .

RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "./"

RUN uv sync --locked

COPY "hw9_infer.py" "./"

ENTRYPOINT [ "uv", "run", "hw9_infer:app"]