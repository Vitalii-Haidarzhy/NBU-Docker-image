import torch
import numpy
from PIL import Image
import torchvision
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("trained_vgg16.ml")
model = model.to(device)
model.eval()
print(model.classifier)

def transform_imagefile(file) -> Image.Image:
    image_row = Image.open(io.BytesIO(file))
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image_row)
    transform1 = transforms.ToPILImage()
    pil_image = transform1(image)
    transform2 = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_tensor = transform2(pil_image).to(device).unsqueeze(0)
    return image_tensor

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/fruits")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = transform_imagefile(await file.read())
    prediction = model(image)
    output = torch.argmax(prediction, 1)
    classes = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']
    return classes[output.item()]
