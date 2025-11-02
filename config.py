import os

class Config:
    """Base configuration"""
    # Get the base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Basic configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Static files configuration
    STATIC_FOLDER = 'static'
    STATIC_URL_PATH = '/static'
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    
    # Ensure absolute paths
    @property
    def DATABASE_PATH(self):
        return os.path.join(self.BASE_DIR, 'emotion_detection.db')
    
    @property
    def UPLOAD_PATH(self):
        return os.path.join(self.BASE_DIR, self.UPLOAD_FOLDER)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    DEVELOPMENT = False
    ENV = 'production'
    # In production, you might want to use a different database path
    DATABASE = os.environ.get('DATABASE_URL', Config.DATABASE)