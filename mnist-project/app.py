from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

model = load_model("mnist_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file)

    image = image.convert("L")

    image = image.resize((28,28))

    image = np.array(image)

    image = image / 255.0

    image = image.reshape(1,28,28,1)

    prediction = model.predict(image)

    digit = int(np.argmax(prediction))

    return {"predicted_digit": digit}