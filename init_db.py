#!/usr/bin/env python3
"""
Database initialization script for Fake News Detector
Run this once to create the database tables.
"""

from app import app, db

def init_db():
    with app.app_context():
        print("Creating database tables...")
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()