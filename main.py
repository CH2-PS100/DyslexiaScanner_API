from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import uvicorn
import os
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from PIL import Image
from io import BytesIO  

app = FastAPI()

# firebase_private_key = os.getenv("FIREBASE_PRIVATE_KEY")
# print("FIREBASE_PRIVATE_KEY:", firebase_private_key)

# # Check if the environment variable is set
# if firebase_private_key is None:
#     raise ValueError("FIREBASE_PRIVATE_KEY environment variable is not set.")


# firebase_private_key = firebase_private_key.replace("\\n", "\n")

# # Create a dictionary with Firebase credentials
# firebase_credentials = {
#     "type": os.getenv("FIREBASE_TYPE"),
#     "project_id": os.getenv("FIREBASE_PROJECT_ID"),
#     "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
#     "private_key": firebase_private_key,
#     "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
#     "client_id": os.getenv("FIREBASE_CLIENT_ID"),
#     "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
#     "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
#     "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
#     "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
#     "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
# }

# cred = credentials.Certificate(firebase_credentials)
# initialize_app(cred)

cred = credentials.Certificate("./credentials.json")
firebase_admin.initialize_app(cred)

# Access Firestore
db = firestore.client()

model_path = os.path.join('models', 'dyslexia_scanner.h5')
loaded_model = tf.keras.models.load_model(model_path)

def preprocess_image(img):
    resized_img = img.convert('RGB').resize((256, 256))

    processed_image = np.array(resized_img) / 255.0

    return np.expand_dims(processed_image, axis=0)


def post_to_firestore(diagnosis, confidence, custom_id=None):
    if custom_id:
        data = {
            "id": int(custom_id),  # Convert custom_id to integer
            "diagnosis": diagnosis,
            "confidence": confidence,
            # Add more fields as needed
        }
        doc_ref = db.collection('dyslexia_data').document(str(custom_id))
        doc_ref.set(data)
        return {"id": int(custom_id), "diagnosis": diagnosis, "confidence": confidence}
    else:

        sequential_id = generate_sequential_id()
        data = {
            "id": sequential_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            # Add more fields as needed
        }
        doc_ref = db.collection('dyslexia_data').document(str(sequential_id))
        doc_ref.set(data)
        return {"id": sequential_id, "diagnosis": diagnosis, "confidence": confidence}

def generate_sequential_id():
    counter = db.collection('counter').document('dyslexia_counter').get().to_dict()
    if counter is None:
        counter = {"value": 1}
        db.collection('counter').document('dyslexia_counter').set(counter)
    else:
        counter["value"] += 1
        db.collection('counter').document('dyslexia_counter').set(counter)
    return counter["value"]

def get_dyslexia_data(doc_id=None):
    dyslexia_data = []

    collection_ref = db.collection('dyslexia_data')

    if doc_id:
        doc_ref = collection_ref.document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  
            dyslexia_data.append(doc_data)
        else:
            return None  # Document not found
    else:
        docs = collection_ref.stream()

        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  
            dyslexia_data.append(doc_data)

    return dyslexia_data


@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))

        processed_image = preprocess_image(img)
        prediction = loaded_model.predict(processed_image)
        prediction_value = prediction[0][0]

        if prediction_value > 0.5:
            diagnosis = "Unfortunately, there is a >50% chance of suffering from dyslexia."
        else:
            diagnosis = "Congratulations, you are normal."

        post_to_firestore(diagnosis, float(prediction_value))
        return {"diagnosis": diagnosis, "confidence": float(prediction_value)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/get_predict')
def get_predict_route():
    try:
        data = get_dyslexia_data()
        return {"dyslexia_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/delete-all")
def delete_all_data():
    try:
        collection_ref = db.collection('dyslexia_data')
        docs = collection_ref.stream()

        for doc in docs:
            doc.reference.delete()

        return {"message": "All dyslexia data deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{doc_id}")
def delete_data_by_id(doc_id: str):
    try:
        # Delete the document with the specified ID from the 'dyslexia_data' collection
        doc_ref = db.collection('dyslexia_data').document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            doc_ref.delete()
            return {"message": f"Dyslexia data with ID {doc_id} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"Dyslexia data with ID {doc_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.put("/update/{doc_id}")
async def update_data_by_id(doc_id: str, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))

        # Preprocess the image
        processed_image = preprocess_image(img)

        # Predict using the loaded model
        prediction = loaded_model.predict(processed_image)
        prediction_value = prediction[0][0]

        doc_ref = db.collection('dyslexia_data').document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            # Update the diagnosis and confidence fields
            doc_ref.update({
                "diagnosis": "Unfortunately, there is a >50% chance of suffering from dyslexia." if prediction_value > 0.5 else "Congratulations, you are normal.",
                "confidence": float(prediction_value),
            })

            return {"message": f"Dyslexia data with ID {doc_id} updated successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"Dyslexia data with ID {doc_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)
