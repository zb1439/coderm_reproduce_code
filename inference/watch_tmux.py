#!/usr/bin/env python3
"""Watch progress of running inference jobs."""
import json, time, os, sys

PROGRESS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "livecodebench", "progress")

def read_progress(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def bar(done, total, w=30):
    pct = done / total if total else 0
    filled = int(pct * w)
    return f"[{'#' * filled}{'.' * (w - filled)}] {done}/{total} ({pct*100:.1f}%)"

def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    while True:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print("=" * 55)
        print(f"  Inference Monitor  |  {time.strftime('%H:%M:%S')}  |  every {interval}s")
        print("=" * 55)

        if not os.path.isdir(PROGRESS_DIR):
            print(f"\n  Progress dir not found: {PROGRESS_DIR}\n")
            time.sleep(interval)
            continue

        files = sorted(f for f in os.listdir(PROGRESS_DIR)
                       if f.startswith("progress_") and f.endswith(".json"))

        if not files:
            print("\n  No progress files found.\n")

        all_done = True
        for fname in files:
            p = read_progress(os.path.join(PROGRESS_DIR, fname))
            if not p:
                continue
            tag = p.get("model_tag", "?")
            status = p.get("status", "?")
            total = p.get("tasks_total", 0)
            done = p.get("tasks_completed", 0)
            cur_id = p.get("current_task_id", "")
            cur_resp = p.get("current_task_responses", 0)
            cur_target = p.get("current_task_target", 100)
            updated = p.get("updated_at_utc", "")[:19]

            if status not in ("completed", "skipped", "failed"):
                all_done = False

            mark = {"running": ">>", "completed": "OK", "failed": "!!", "skipped": "--"}.get(status, "??")
            print(f"\n  [{mark}] {tag}")
            print(f"       Tasks: {bar(done, total)}")
            if status == "running":
                print(f"       Now:   task {cur_id}  {bar(cur_resp, cur_target, 20)}")
            else:
                print(f"       Status: {status}")
            print(f"       Updated: {updated}")

        print(f"\n{'-' * 55}")
        print("  Ctrl+C to exit")

        if all_done and files:
            print("\n  All jobs finished!")
            break

        time.sleep(interval)

if __name__ == "__main__":
    main()
