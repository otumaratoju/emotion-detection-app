# Emotion Detection Web Application

This is a web application that uses deep learning to detect emotions from images and live video captures. The application is built with Flask and TensorFlow, and it can detect seven different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Image upload for emotion detection
- Live video capture for real-time emotion detection
- User management system
- Emotion detection history tracking
- Statistical analysis of emotions
- Responsive web design
- SQLite database integration

## Technical Stack

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Database: SQLite
- Machine Learning: TensorFlow, OpenCV
- Deployment: Render.com

## Project Structure

```
emotion_detection_webapp/
├── static/
│   ├── style.css
│   └── uploads/
├── templates/
│   └── index.html
├── app.py
├── model.py
├── requirements.txt
├── emotion_model.h5
├── emotion_detection.db
├── link_to_my_web_app.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_detection_webapp
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. You can:
   - Enter your name for tracking your emotion detection history
   - Upload an image for emotion detection
   - Use your webcam for real-time emotion detection
   - View your detection history
   - See emotion detection statistics

## API Endpoints

- `GET /`: Main page
- `POST /upload`: Upload and process an image
- `GET /video`: Stream video for real-time detection
- `GET /history`: Get user's detection history
- `GET /statistics`: Get emotion detection statistics

## Database Schema

### Users Table
- id (PRIMARY KEY)
- username
- is_online
- created_at

### Emotion Results Table
- id (PRIMARY KEY)
- user_id (FOREIGN KEY)
- image_path
- emotion
- confidence
- created_at

## Model Information

The emotion detection model is a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It:
- Takes 48x48 grayscale images as input
- Detects 7 different emotions
- Uses multiple convolutional and pooling layers
- Includes dropout layers for regularization
- Provides confidence scores for predictions

## Deployment

The application is deployed on Render.com. You can access it at:
[https://emotion-detection-webapp.onrender.com](https://emotion-detection-webapp.onrender.com)

## Future Improvements

- Add user authentication
- Implement emotion tracking over time
- Add more sophisticated emotion detection models
- Include batch processing capabilities
- Add export functionality for detection results

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.