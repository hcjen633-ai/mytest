# api/index.py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Python on Vercel!"

@app.route('/about')
def about():
    return "This is the about page."