import schedule
import time
import subprocess

def retrain_model():
    print("Starting retraining process...")
    subprocess.run(["python", "retrain_model.py"])
    print("Retraining process completed.")

# Schedule the retraining process to run twice a week.
schedule.every().saturday.at("12:00").do(retrain_model)
schedule.every().wednesday.at("11:05").do(retrain_model)

# Keep the script running and checking for scheduled tasks.
while True:
    schedule.run_pending()
    time.sleep(60)
