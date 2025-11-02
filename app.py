from flask import Flask, render_template, request, jsonify, Response, url_for
import os
from werkzeug.utils import secure_filename
import sqlite3
import cv2
from datetime import datetime
from model import EmotionDetector
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_SUBFOLDER = 'uploads'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, UPLOAD_SUBFOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize detector
emotion_detector = EmotionDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_online BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            emotion TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


@app.after_request
def apply_cors(response):
    # allow developer browser requests; harmless in production
    response.headers.setdefault('Cache-Control', 'no-store')
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.route('/')
def home():
    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init failed: {e}")
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_image():
    # Handle preflight
    if request.method == 'OPTIONS':
        return ('', 204)

    # Accepts a file named 'image' (either user-uploaded file input or camera blob)
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['image']
    username = request.form.get('username', 'Anonymous')

    if file.filename == '' and getattr(file, 'stream', None) is None:
        return jsonify({'error': 'No selected file'}), 400

    # Some camera blobs may come without a filename; ensure secure name
    orig_filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    ext = orig_filename.rsplit('.', 1)[1].lower() if '.' in orig_filename else 'jpg'
    if ext not in ALLOWED_EXTENSIONS:
        # allow camera blob which we will treat as jpg
        if file.filename and ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{orig_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        logger.info(f"Saved upload to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    # Run emotion detection
    try:
        result = emotion_detector.detect_emotion(save_path)
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return jsonify({'error': 'Emotion detection failed'}), 500

    if not result or 'emotion' not in result:
        return jsonify({'error': 'No emotion detected'}), 500

    # Store (best-effort)
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO users (username) VALUES (?)', (username,))
        cursor.execute('SELECT id FROM users WHERE username=?', (username,))
        user_id = cursor.fetchone()[0]
        cursor.execute('INSERT INTO emotion_results (user_id, image_path, emotion, confidence) VALUES (?,?,?,?)',
                       (user_id, save_path, result['emotion'], result.get('confidence', 0)))
        conn.commit()
    except Exception as e:
        logger.warning(f"DB write failed (non-fatal): {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    image_url = url_for('static', filename=f'{UPLOAD_SUBFOLDER}/{filename}')
    return jsonify({
        'success': True,
        'emotion': result['emotion'],
        'confidence': result.get('confidence', 0),
        'image_url': image_url
    })


def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error('Could not open camera')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f'Frame processing error: {e}')
                continue
    finally:
        camera.release()


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def get_history():
    username = request.args.get('username', 'Anonymous')
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('''SELECT er.created_at, er.emotion, er.confidence, er.image_path
                          FROM emotion_results er JOIN users u ON er.user_id = u.id
                          WHERE u.username = ? ORDER BY er.created_at DESC LIMIT 10''', (username,))
        rows = cursor.fetchall()
        history = [{'timestamp': r[0], 'emotion': r[1], 'confidence': r[2], 'image_url': url_for('static', filename=f'{UPLOAD_SUBFOLDER}/' + os.path.basename(r[3]))} for r in rows]
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        logger.error(f'History read error: {e}')
        return jsonify({'error': 'Failed to read history'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.route('/statistics')
def get_statistics():
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('SELECT emotion, COUNT(*) FROM emotion_results GROUP BY emotion')
        distribution = dict(cursor.fetchall())
        cursor.execute('SELECT COUNT(*) FROM emotion_results')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM emotion_results')
        users = cursor.fetchone()[0]
        return jsonify({'success': True, 'emotion_distribution': distribution, 'total_detections': total, 'total_users': users})
    except Exception as e:
        logger.error(f'Statistics error: {e}')
        return jsonify({'error': 'Failed to get statistics'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
from flask import Flask, render_template, request, jsonify, Response, url_for
import os
from werkzeug.utils import secure_filename
import sqlite3
import cv2
from datetime import datetime
from model import EmotionDetector
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_SUBFOLDER = 'uploads'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, UPLOAD_SUBFOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize detector
emotion_detector = EmotionDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_online BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            emotion TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


@app.after_request
def apply_cors(response):
    # allow developer browser requests; harmless in production
    response.headers.setdefault('Cache-Control', 'no-store')
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/')
def home():
    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init failed: {e}")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # Accepts a file named 'image' (either user-uploaded file input or camera blob)
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['image']
    username = request.form.get('username', 'Anonymous')

    if file.filename == '' and getattr(file, 'stream', None) is None:
        return jsonify({'error': 'No selected file'}), 400

    # Some camera blobs may come without a filename; ensure secure name
    orig_filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    ext = orig_filename.rsplit('.', 1)[1].lower() if '.' in orig_filename else 'jpg'
    if ext not in ALLOWED_EXTENSIONS:
        # allow camera blob which we will treat as jpg
        if file.filename and ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{orig_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        logger.info(f"Saved upload to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    # Run emotion detection
    try:
        result = emotion_detector.detect_emotion(save_path)
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return jsonify({'error': 'Emotion detection failed'}), 500

    if not result or 'emotion' not in result:
        return jsonify({'error': 'No emotion detected'}), 500

    # Store (best-effort)
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO users (username) VALUES (?)', (username,))
        cursor.execute('SELECT id FROM users WHERE username=?', (username,))
        user_id = cursor.fetchone()[0]
        cursor.execute('INSERT INTO emotion_results (user_id, image_path, emotion, confidence) VALUES (?,?,?,?)',
                       (user_id, save_path, result['emotion'], result.get('confidence', 0)))
        conn.commit()
    except Exception as e:
        logger.warning(f"DB write failed (non-fatal): {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    image_url = url_for('static', filename=f'{UPLOAD_SUBFOLDER}/{filename}')
    return jsonify({
        'success': True,
        'emotion': result['emotion'],
        'confidence': result.get('confidence', 0),
        'image_url': image_url
    })


@app.route('/upload', methods=['OPTIONS'])
def upload_options():
    # Reply to CORS preflight requests
    return ('', 204)


def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error('Could not open camera')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f'Frame processing error: {e}')
                continue
    finally:
        camera.release()


from flask import Flask, render_template, request, jsonify, Response, url_for
import os
from werkzeug.utils import secure_filename
import sqlite3
import cv2
from datetime import datetime
from model import EmotionDetector
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_SUBFOLDER = 'uploads'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, UPLOAD_SUBFOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize detector
emotion_detector = EmotionDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_online BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            emotion TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


@app.after_request
def apply_cors(response):
    # allow developer browser requests; harmless in production
    response.headers.setdefault('Cache-Control', 'no-store')
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/')
def home():
    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init failed: {e}")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # Accepts a file named 'image' (either user-uploaded file input or camera blob)
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['image']
    username = request.form.get('username', 'Anonymous')

    if file.filename == '' and getattr(file, 'stream', None) is None:
        return jsonify({'error': 'No selected file'}), 400

    # Some camera blobs may come without a filename; ensure secure name
    orig_filename = secure_filename(file.filename) if file.filename else 'capture.jpg'
    ext = orig_filename.rsplit('.', 1)[1].lower() if '.' in orig_filename else 'jpg'
    if ext not in ALLOWED_EXTENSIONS:
        # allow camera blob which we will treat as jpg
        if file.filename and ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{orig_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        logger.info(f"Saved upload to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    # Run emotion detection
    try:
        result = emotion_detector.detect_emotion(save_path)
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return jsonify({'error': 'Emotion detection failed'}), 500

    if not result or 'emotion' not in result:
        return jsonify({'error': 'No emotion detected'}), 500

    # Store (best-effort)
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO users (username) VALUES (?)', (username,))
        cursor.execute('SELECT id FROM users WHERE username=?', (username,))
        user_id = cursor.fetchone()[0]
        cursor.execute('INSERT INTO emotion_results (user_id, image_path, emotion, confidence) VALUES (?,?,?,?)',
                       (user_id, save_path, result['emotion'], result.get('confidence', 0)))
        conn.commit()
    except Exception as e:
        logger.warning(f"DB write failed (non-fatal): {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    image_url = url_for('static', filename=f'{UPLOAD_SUBFOLDER}/{filename}')
    return jsonify({
        'success': True,
        'emotion': result['emotion'],
        'confidence': result.get('confidence', 0),
        'image_url': image_url
    })


def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error('Could not open camera')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f'Frame processing error: {e}')
                continue
    finally:
        camera.release()


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def get_history():
    username = request.args.get('username', 'Anonymous')
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('''SELECT er.created_at, er.emotion, er.confidence, er.image_path
                          FROM emotion_results er JOIN users u ON er.user_id = u.id
                          WHERE u.username = ? ORDER BY er.created_at DESC LIMIT 10''', (username,))
        rows = cursor.fetchall()
        history = [{'timestamp': r[0], 'emotion': r[1], 'confidence': r[2], 'image_url': url_for('static', filename=f'{UPLOAD_SUBFOLDER}/' + os.path.basename(r[3]))} for r in rows]
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        logger.error(f'History read error: {e}')
        return jsonify({'error': 'Failed to read history'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.route('/statistics')
def get_statistics():
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'emotion_detection.db'))
        cursor = conn.cursor()
        cursor.execute('SELECT emotion, COUNT(*) FROM emotion_results GROUP BY emotion')
        distribution = dict(cursor.fetchall())
        cursor.execute('SELECT COUNT(*) FROM emotion_results')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM emotion_results')
        users = cursor.fetchone()[0]
        return jsonify({'success': True, 'emotion_distribution': distribution, 'total_detections': total, 'total_users': users})
    except Exception as e:
        logger.error(f'Statistics error: {e}')
        return jsonify({'error': 'Failed to get statistics'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

