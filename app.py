from flask import Flask, render_template, request
import polibot_ref

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lsm')
def lsm():
    return render_template('lsm.html')

@app.route('/torniquete')
def torniquete():
    return render_template('torniquete.html')

@app.route('/chat', methods=['POST'])
def chat():
    # AI logic from polibot_ref.py
    pass

if __name__ == '__main__':
    app.run(debug=True)
