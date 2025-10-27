import smtplib
import random
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- In-memory storage for OTPs (for simplicity) ---
# In a production system, you would use a database like Redis for this.
otp_storage = {}

def generate_otp(email: str):
    """Generates a 6-digit OTP and stores it."""
    otp = str(random.randint(100000, 999999))
    otp_storage[email] = otp
    print(f"Generated OTP for {email}: {otp}") # For debugging purposes
    return otp

def verify_otp(email: str, received_otp: str):
    """Verifies the received OTP against the stored one."""
    stored_otp = otp_storage.get(email)
    if stored_otp and stored_otp == received_otp:
        # OTP is correct, remove it after verification
        del otp_storage[email]
        return True
    return False

def send_otp_email(recipient_email: str, otp: str):
    """Sends the OTP to the user's email address using Gmail."""
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")

    if not email_address or not email_password:
        print("ERROR: Email credentials are not set in the .env file.")
        return False

    try:
        # Set up the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your Verification Code for AI Lesion Analyzer"
        message["From"] = f"AI Lesion Analyzer <{email_address}>"
        message["To"] = recipient_email

        html_content = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; text-align: center; color: #333;">
                <h2>Verification Required</h2>
                <p>Your one-time password for the AI-Augmented Skin Lesion Analyzer is:</p>
                <p style="font-size: 24px; font-weight: bold; letter-spacing: 2px; color: #2d3748; background-color: #f7fafc; border: 1px solid #e2e8f0; padding: 10px 20px; border-radius: 8px; display: inline-block;">
                    {otp}
                </p>
                <p>This code will expire in 10 minutes. If you did not request this, please ignore this email.</p>
            </div>
        </body>
        </html>
        """
        
        message.attach(MIMEText(html_content, "html"))

        # Connect to Gmail's SMTP server and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_address, email_password)
            server.sendmail(email_address, recipient_email, message.as_string())
        
        print(f"Successfully sent OTP to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
