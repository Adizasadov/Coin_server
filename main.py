import torch

print("Hello World")
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1><a href='https://google.de'>Google</a>"

if __name__ == '__main__':
    app.run(debug=True)
    