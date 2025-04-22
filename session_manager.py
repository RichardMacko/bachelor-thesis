import os
from instagrapi import Client
from pathlib import Path

SESSION_FILE = Path("session.json")


class SessionManager:
    def __init__(self, client: Client):
        self.client = client

    def save_session(self):
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.client.dump_settings(SESSION_FILE)

    def load_session(self):
        if SESSION_FILE.exists():
            try:
                self.client.load_settings(SESSION_FILE)
                print("Session loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading session: {e}")
                return False
        else:
            self.save_session()
            return False

