import os
# os.system('sh setup.sh')
from flask import Flask, render_template, request, send_from_directory

# from main import main

app = Flask(__name__, static_url_path='/static')

# output_name = 'static/new.mp3'

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    output_name = 'static/new.mp3'
    sound_file=request.files['soundfile']
    sound_path = './static/'+sound_file.filename
    sound_file.save(sound_path)
    # main(sound_path, output_name)
    if not os.path.exists(output_name):
        output_name = sound_path
    return render_template('index.html', output=output_name)

# @app.route('/js/<path:path>')
# def send_js(path):
#     return send_from_directory('js', path)

# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


if __name__=="__main__":
    app.run(debug=True)#, port=3000, host='0.0.0.0')