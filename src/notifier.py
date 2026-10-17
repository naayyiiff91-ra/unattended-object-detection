# Windows Native Desktop Toast Notification Dispatcher
import os
import threading


def send_alert_notification(bag_id, class_name, alert_text, snapshot_path=None):
    """
    Dispatches a native Windows 10/11 desktop toast notification banner
    with audio chime and snapshot preview icon.
    Executes in a daemon thread so it never interrupts the YOLO video tracking loop.
    """
    def _notify():
        try:
            from winotify import Notification, audio

            title = f"SECURITY ALERT: Unattended {class_name.upper()} #{bag_id}"
            clean_msg = alert_text.strip() if alert_text else f"An unattended {class_name} was detected."

            icon_path = ""
            if snapshot_path and os.path.exists(snapshot_path):
                icon_path = os.path.abspath(snapshot_path)

            toast = Notification(
                app_id="SentinAI Security Monitor",
                title=title,
                msg=clean_msg,
                icon=icon_path
            )
            # Use Windows alert sound
            toast.set_audio(audio.Mail, loop=False)
            toast.show()
            print(f"[NOTIFICATION] Desktop toast alert displayed for Bag #{bag_id}")

        except Exception as err:
            print(f"[NOTIFICATION WARNING] Could not display toast notification: {err}")

    # Dispatch asynchronously
    threading.Thread(target=_notify, daemon=True).start()


if __name__ == "__main__":
    print("Testing desktop notification...")
    send_alert_notification(
        bag_id=99,
        class_name="backpack",
        alert_text="Test Alert: A black backpack was left unattended for 15 seconds."
    )

