FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "./"

RUN uv sync --locked

COPY "fapi.py" "./"

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "fapi:app", "--host", "0.0.0.0", "--port", "9696" ]