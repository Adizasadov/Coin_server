import torch

print("Hello World")
from flask import Flask, request

app = Flask(__name__)

@app.route('/example', methods=['POST'])
def hello_world():
    binary_data = request.data
    with open('uploaded_file.bin', 'wb') as f:
            f.write(binary_data)
    return "<h1>Hello, World!</h1><a href='https://google.de'>Google</a>"

if __name__ == '__main__':
    app.run(debug=True)
    
