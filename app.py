from flask import Flask, render_template, request, jsonify, Response
import os
from werkzeug.utils import secure_filename
import sqlite3
import cv2
import numpy as np
from datetime import datetime
from model import EmotionDetector

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize emotion detector
emotion_detector = EmotionDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            is_online BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create emotion_results table
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

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and emotion detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    username = request.form.get('username', 'Anonymous')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform emotion detection
            result = emotion_detector.detect_emotion(filepath)
            
            # Store the result in database
            conn = sqlite3.connect('emotion_detection.db')
            cursor = conn.cursor()
            
            # Insert or update user
            cursor.execute('INSERT OR IGNORE INTO users (username) VALUES (?)', (username,))
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user_id = cursor.fetchone()[0]
            
            # Store detection result
            cursor.execute('''
                INSERT INTO emotion_results (user_id, image_path, emotion, confidence)
                VALUES (?, ?, ?, ?)
            ''', (user_id, filepath, result['emotion'], result['confidence']))
            
            conn.commit()
            conn.close()
            
            # Return result with image URL
            result['image_url'] = f'/static/uploads/{filename}'
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video')
def video():
    """Handle live video streaming"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with emotion detection"""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        try:
            # Convert frame to format suitable for emotion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect face in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Get face region and detect emotion
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                result = emotion_detector.detect_emotion_from_array(face)
                
                # Display emotion text
                text = f"{result['emotion']} ({result['confidence']:.2f})"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/history')
def get_history():
    """Get emotion detection history for a user"""
    username = request.args.get('username', 'Anonymous')
    
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT er.created_at, er.emotion, er.confidence, er.image_path
        FROM emotion_results er
        JOIN users u ON er.user_id = u.id
        WHERE u.username = ?
        ORDER BY er.created_at DESC
        LIMIT 10
    ''', (username,))
    
    history = [{
        'timestamp': row[0],
        'emotion': row[1],
        'confidence': row[2],
        'image_path': row[3]
    } for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(history)

@app.route('/statistics')
def get_statistics():
    """Get emotion detection statistics"""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    
    # Get emotion distribution
    cursor.execute('''
        SELECT emotion, COUNT(*) as count
        FROM emotion_results
        GROUP BY emotion
        ORDER BY count DESC
    ''')
    
    stats = {
        'emotion_distribution': dict(cursor.fetchall()),
        'total_detections': cursor.execute('SELECT COUNT(*) FROM emotion_results').fetchone()[0],
        'total_users': cursor.execute('SELECT COUNT(DISTINCT user_id) FROM emotion_results').fetchone()[0]
    }
    
    conn.close()
    return jsonify(stats)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)