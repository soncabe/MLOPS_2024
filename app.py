from fastapi import FastAPI, UploadFile, File
import uvicorn
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing import image


app = FastAPI()
model = pickle.load(open("CAD_CNN.pkl", "rb"))


@app.get("/")
async def root():
    return {"message": "Cat and Dog Classification with CNN"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": f"Successfully uploaded {file.filename}"}
    finally:
        file.file.close()

    test_image = image.load_img(file.filename, target_size = (128,128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    
    return {
            f"La foto subida es un {prediction}"
        }



if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
