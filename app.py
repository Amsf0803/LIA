from flask import request, redirect, Flask, Response, render_template, jsonify, url_for, flash, session




app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = 'clave_secreta_segura'


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/lsm')
def lsm():
    return render_template('lsm.html')

@app.route('/torniquete')
def torniquete():
    return render_template('torniquete.html')   



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
