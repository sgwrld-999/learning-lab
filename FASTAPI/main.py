from fastapi import FastAPI

app = FastAPI()

@app.get("check1")
def check1():
    return {message: "you call check1 hit point"}

 @app.get("check2")
def check1():
    return {message: "you call check2 hit point"}