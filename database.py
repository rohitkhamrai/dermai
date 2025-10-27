import sqlite3
import json
from datetime import datetime

DATABASE_NAME = "analysis_history.db"

def init_db():
    """Initializes the database and creates the 'analyses' table if it doesn't exist."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                image_filename TEXT,
                patient_age INTEGER,
                patient_sex TEXT,
                lesion_location TEXT,
                final_report TEXT,
                heatmap_image_base64 TEXT,
                raw_data_json TEXT,
                corrected_label TEXT
            )
        """)
        conn.commit()
    print("Database initialized successfully.")

def add_analysis(user_email, image_filename, patient_data, analysis_result):
    """Adds a new analysis record to the database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    raw_data_str = json.dumps(analysis_result.get("raw_data", {}))
    
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (
                user_email, analysis_timestamp, image_filename, patient_age, 
                patient_sex, lesion_location, final_report, heatmap_image_base64, 
                raw_data_json, corrected_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """, (
            user_email, timestamp, image_filename, patient_data['age'], 
            patient_data['sex'], patient_data['localization'], 
            analysis_result.get("final_report"), 
            analysis_result.get("heatmap_image"), 
            raw_data_str
        ))
        conn.commit()
        # Return the complete record that was just inserted
        last_id = cursor.lastrowid
        return get_analysis_by_id(last_id)


def get_analyses_by_email(user_email: str):
    """Retrieves all analysis records for a given user, ordered by most recent."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE user_email = ? ORDER BY id DESC", (user_email,))
        rows = cursor.fetchall()
        # Convert rows to a list of dictionaries for easy JSON serialization
        return [dict(row) for row in rows]

def update_correction(analysis_id: int, corrected_label: str, user_email: str):
    """Updates the corrected_label for a specific analysis, ensuring user owns it."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        # The WHERE clause ensures a user can only update their own records
        cursor.execute(
            "UPDATE analyses SET corrected_label = ? WHERE id = ? AND user_email = ?",
            (corrected_label, analysis_id, user_email)
        )
        conn.commit()
        return cursor.rowcount > 0 # Returns True if a row was updated, False otherwise

def get_analysis_by_id(analysis_id: int):
    """Retrieves a single analysis record by its ID."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

