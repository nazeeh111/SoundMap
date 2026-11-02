from flask import Flask, render_template, Response, abort
from bokeh.embed import server_document
import argparse
import json
import os

app = Flask(__name__, template_folder=os.path.abspath('templates'))
app.config['CAMERA_ENABLED'] = False


@app.route('/video_feed')
def video_feed():
    if not app.config['CAMERA_ENABLED']:
        abort(403, description='Camera disabled. Restart with --camera only when capture is authorized.')
    import cv2

    def gen_frames():
        cap = cv2.VideoCapture(0)
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                encoded, buffer = cv2.imencode('.jpg', frame)
                if not encoded:
                    break
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        finally:
            cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    bokeh_script = server_document('http://localhost:5006/bokeh_app')
    with open('config/config.json') as file:
        config = json.load(file)
    return render_template('index.html', bokeh_script=bokeh_script, json_data=config,
                           camera_enabled=app.config['CAMERA_ENABLED'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    args = parser.parse_args()
    app.config['CAMERA_ENABLED'] = args.camera
    app.run(host='127.0.0.1', port=5000, debug=False)
