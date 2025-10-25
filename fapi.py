import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


class ClientInput(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


@app.post("/predict")
def predict_sub(client: ClientInput):
    print(client)
    result = model.predict_proba(dv.transform(client.__dict__))
    print("test: ", result)
    return result.tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
