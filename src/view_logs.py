# a standalone script to read and display all logged events in a clean table.
import sqlite3
import os

DB_PATH = "logs/events.db"


def view_logs():
    if not os.path.exists(DB_PATH):
        print("No events database found yet. Run main.py first and let a bag go unattended.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message
        FROM events
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No events logged yet.")
        return

    # Column headers
    headers = ["ID", "Bag ID", "Type", "Duration (s)", "Timestamp", "Snapshot", "Alert Message"]

    # Calculate column widths based on content
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    def print_row(values):
        line = " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(values))
        print(line)

    def print_separator():
        print("-+-".join("-" * w for w in col_widths))

    print(f"\nTotal events logged: {len(rows)}\n")
    print_row(headers)
    print_separator()

    for row in rows:
        # Round duration for readability
        row = list(row)
        row[3] = round(row[3], 1)
        print_row(row)

    print()


if __name__ == "__main__":
    view_logs()