# db.py
import sqlite3

def init_db(db_path="logs.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Ensure TrafficLog table exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS TrafficLog (
            time TEXT,
            camera_id TEXT,
            count_in INTEGER,
            count_out INTEGER,
            posture TEXT,
            alert TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_log(conn, log_entry):
    c = conn.cursor()
    # Ensure table exists before insert
    c.execute('''
        CREATE TABLE IF NOT EXISTS TrafficLog (
            time TEXT,
            camera_id TEXT,
            count_in INTEGER,
            count_out INTEGER,
            posture TEXT,
            alert TEXT
        )
    ''')
    c.execute('''
        INSERT INTO TrafficLog (time, camera_id, count_in, count_out, posture, alert)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        log_entry["time"],
        log_entry["camera_id"],
        log_entry["in"],
        log_entry["out"],
        log_entry["posture"],
        log_entry["alert"]
    ))
    conn.commit()
