from flask import Flask, render_template, Response, request
import cv2
import subprocess
import os
from PIL import Image
from io import BytesIO
import glob

app = Flask(__name__)

# Path to your YOLO model and other configurations
weights_path = 'runs/train/exp2/weights/best.pt'
result_dir = 'runs/detect'

def run_detection(filepath):
    detect_command = (
        f"python detect.py --weights {weights_path} "
        f"--img 640 --conf 0.25 --source {filepath}"
    )
    subprocess.run(detect_command, shell=True, check=True)

def get_latest_result_image():
    list_of_files = glob.glob(os.path.join(result_dir, 'exp*', '*.jpg'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        run_detection(filepath)
        result_image_path = get_latest_result_image()
        if result_image_path:
            return send_file(result_image_path, mimetype='image/jpeg')
        else:
            return "Detection failed"

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        camera = cv2.VideoCapture(1)  # Use the default camera (change the index if you have multiple cameras)
        temp_image_path = 'temp.jpg'

        while True:
            success, frame = camera.read()  # Read the frame from the camera

            if not success:
                break
            else:
                frame = cv2.resize(frame, (640, 480))  # Resize the frame for faster processing
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # Save frame as an image to run detection
                cv2.imwrite(temp_image_path, frame)
                run_detection(temp_image_path)  # Run detection on the frame

                # Load and return the result image
                result_image_path = get_latest_result_image()
                if result_image_path and os.path.exists(result_image_path):
                    result_image = Image.open(result_image_path)
                    result_image = result_image.convert("RGB")
                    buffer = BytesIO()
                    result_image.save(buffer, format="JPEG")
                    frame = buffer.getvalue()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    break

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
