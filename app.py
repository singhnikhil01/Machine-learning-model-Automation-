from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI()


# can get items
class Item(BaseModel):
    name: str
    price: float
    quantity: int


# can get arrays from the body
# can get other commands from the https://fastapi.tiangolo.com/tutorial/body/


@app.post("/items/")
async def create_item(item: Item):
    # Process the item and return a response
    return {"message": "Item created successfully"}


# if user ask for base url then return this
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file.file)

        # Process the DataFrame as needed (you can perform various operations here)

        return JSONResponse(
            content={
                "message": "File uploaded successfully",
                "columns": list(df.columns),
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing file: {str(e)}"}, status_code=500
        )
