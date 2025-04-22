from instagrapi.exceptions import (
    BadPassword, ChallengeRequired, FeedbackRequired,
    PleaseWaitFewMinutes, LoginRequired, PrivateAccount
)


class ExceptionHandler:
    def __init__(self, client):
        self.client = client

    def handle_exception(self, e):
        if isinstance(e, BadPassword):
            print("Incorrect password. Please check your credentials.")
            return False

        elif isinstance(e, LoginRequired):
            print("Instagram requires re-login!")
            try:
                self.client.relogin()
                print("Logged in successfully.")
            except Exception as relogin_error:
                print(f"Error during relogin: {relogin_error}")
                username = input("Username: ")
                passw = input("Password: ")
                try:
                    self.client.login(username, passw)
                    print("Manual login successful!")
                except Exception as manual_login_error:
                    print(f"Error during manual login: {manual_login_error}")

        elif isinstance(e, ChallengeRequired):
            print("Instagram requires challenge.")
            check = input("After solving the challenge, type ok: ")
            if check == "ok":
                try:
                    self.client.relogin()
                    print("Continuing after challenge.")
                except Exception as challenge_relogin_error:
                    print(f"Error after re-doing challenge: {challenge_relogin_error}")

        elif isinstance(e, FeedbackRequired):
            print("Instagram requires feedback")

        elif isinstance(e, PleaseWaitFewMinutes):
            print("Please wait few minutes before trying again.")

        elif isinstance(e, PrivateAccount):
            print("This account is private")

        else:
            print(f"An unexpected error occurred")
