from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time

# Define class labels
class_labels = {
    0: 'Adipose (ADI)',
    1: 'background (BACK)',
    2: 'debris (DEB)',
    3: 'lymphocytes (LYM)',
    4: 'mucus (MUC)',
    5: 'smooth muscle (MUS)',
    6: 'normal colon mucosa (NORM)',
    7: 'cancer-associated stroma (STR)',
    8: 'colorectal adenocarcinoma epithelium (TUM)'
}

# Load the model
model = load_model('models/vgg19_bs100_e10.h5')

app = FastAPI()

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize as per your model requirements
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.  # Normalize pixel values
    return img_array

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        img_array = preprocess_image(img)

        # Record start time
        start_time = time.time()

        # Perform inference
        predictions = model.predict(img_array)

        # Record end time
        end_time = time.time()

        # Calculate inference time
        inference_time = end_time - start_time

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)

        # Get the corresponding class name and probability
        predicted_class_name = class_labels[predicted_class_index]
        predicted_probability = float(predictions[0][predicted_class_index])

        return JSONResponse(content={"inference_time": inference_time, 
                                     "predicted_class": predicted_class_name,
                                     "predicted_probability": predicted_probability})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
