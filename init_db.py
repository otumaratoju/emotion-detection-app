import sqlite3
import os
from datetime import datetime
from config import DevelopmentConfig, ProductionConfig

def init_db(config_class=DevelopmentConfig):
    """Initialize the database with required tables"""
    try:
        conn = sqlite3.connect(config_class.DATABASE)
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
        print("Database initialized successfully!")
        
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    # Initialize database based on environment
    if os.environ.get('FLASK_ENV') == 'production':
        init_db(ProductionConfig)
    else:
        init_db(DevelopmentConfig)