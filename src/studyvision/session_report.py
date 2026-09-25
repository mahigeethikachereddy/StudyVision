"""Save live-session observations and build a local HTML summary dashboard."""

import csv
import html
from collections import Counter
from datetime import datetime
from pathlib import Path


class SessionLogger:
    """Record one low-frequency observation per live prediction session."""

    def __init__(self, started_at, sample_interval=1.0):
        self.started_at = started_at
        self.sample_interval = sample_interval
        self.rows = []
        self._last_recorded_at = None

    def record(self, timestamp, state, activity="", confidence=None):
        if self._last_recorded_at is not None and timestamp - self._last_recorded_at < self.sample_interval:
            return
        self.rows.append({
            "elapsed_seconds": round(timestamp - self.started_at, 1),
            "state": state,
            "activity": activity,
            "confidence": "" if confidence is None else round(float(confidence), 4),
        })
        self._last_recorded_at = timestamp

    def save(self, project_root):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = project_root / "data" / "session_logs"
        reports_dir = project_root / "reports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = logs_dir / f"session_{stamp}.csv"
        report_path = reports_dir / f"session_{stamp}_summary.html"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["elapsed_seconds", "state", "activity", "confidence"])
            writer.writeheader()
            writer.writerows(self.rows)
        report_path.write_text(build_dashboard(self.rows, stamp), encoding="utf-8")
        return csv_path, report_path


def build_dashboard(rows, session_id):
    """Return a self-contained dashboard for the recorded observations."""
    total = len(rows)
    state_counts = Counter(row["state"] for row in rows)
    duration = rows[-1]["elapsed_seconds"] if rows else 0
    studying = state_counts["studying"]
    distracted = state_counts["distracted"]
    break_time = state_counts["on_break"]
    distraction_episodes = _episodes(rows, "distracted")
    break_episodes = _episodes(rows, "on_break")
    feedback, suggestions = _feedback(
        studying + distracted,
        studying,
        distracted,
        distraction_episodes,
    )
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>StudyVision Session Summary</title><style>
body{{max-width:920px;margin:40px auto;padding:0 20px;font-family:Arial,sans-serif;color:#172033;background:#f5f7fb}}h1{{margin-bottom:4px}}.muted{{color:#667085}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}}.card,section{{background:white;border:1px solid #dbe2ea;border-radius:10px;padding:18px}}.value{{display:block;font-size:28px;font-weight:700;margin-top:8px}}section{{margin-top:16px}}.bar-row{{margin:14px 0}}.bar-label{{display:flex;justify-content:space-between;margin-bottom:6px}}.bar{{height:10px;background:#e8edf3;border-radius:5px;overflow:hidden}}.fill{{height:100%;background:#3168c6}}.note{{border-left:4px solid #c48a16;padding:12px;background:#fff9eb}}
</style></head><body><h1>StudyVision Session Summary</h1><p class=\"muted\">Session {html.escape(session_id)} · observations sampled once per second</p>
<div class=\"grid\"><div class=\"card\">Session duration<span class=\"value\">{duration:.0f}s</span></div><div class=\"card\">Observed studying<span class=\"value\">{studying}s</span></div><div class=\"card\">Breaks taken<span class=\"value\">{break_episodes}</span></div><div class=\"card\">Manual break time<span class=\"value\">{break_time}s</span></div><div class=\"card\">Distraction episodes<span class=\"value\">{distraction_episodes}</span></div><div class=\"card\">Observed distracted<span class=\"value\">{distracted}s</span></div></div>
<section><h2>Studying, breaks, and distraction throughout the session</h2>{_bars(state_counts,total,{"studying":"Studying","on_break":"Manual break","distracted":"Distracted"})}</section>
<section><h2>Your session summary</h2><p>{html.escape(feedback)}</p><h3>For your next session</h3><ul>{suggestions}</ul></section>
<section class=\"note\"><strong>Interpretation:</strong> “Manual break” is recorded when the student presses B. “Distracted” is recorded only after the face has been continuously absent from the webcam for three seconds. These are observable session signals, not claims about a student's mental state.</section></body></html>"""


def _percent(part, total):
    return "—" if not total else f"{part / total:.0%}"


def _bars(counts, total, labels):
    if not total:
        return "<p class=\"muted\">No observations were recorded.</p>"
    items = []
    for key, count in counts.most_common():
        label = labels.get(key, key) if labels else key.replace("_", " ")
        percentage = count / total * 100
        items.append(f'<div class="bar-row"><div class="bar-label"><span>{html.escape(label)}</span><span>{percentage:.0f}% ({count}s)</span></div><div class="bar"><div class="fill" style="width:{percentage:.1f}%"></div></div></div>')
    return "".join(items)


def _episodes(rows, target_state):
    """Count transitions into a state rather than sampling seconds."""
    previous_state = None
    count = 0
    for row in rows:
        if row["state"] == target_state and previous_state != target_state:
            count += 1
        previous_state = row["state"]
    return count


def _feedback(total, studying, distracted, distraction_episodes):
    """Generate constructive feedback from observable session signals."""
    if not total:
        return (
            "No observations were recorded, so there is not enough information to summarise this session.",
            "<li>Keep the webcam view clear and try another session.</li>",
        )

    distracted_ratio = distracted / total

    if distracted == 0:
        return (
            "You stayed visible throughout the observed session. That is a strong sign of a consistent study setup.",
            "<li>Keep the same environment and take planned breaks between sessions.</li>"
            "<li>Try a slightly longer session when you feel ready.</li>",
        )

    if distracted_ratio <= 0.10:
        return (
            "You were visible for most of the session, with only brief interruptions.",
            "<li>Keep your study materials ready before you begin.</li>"
            "<li>Make breaks intentional: finish a small task before stepping away.</li>",
        )

    if distracted_ratio <= 0.25:
        return (
            f"You returned to studying after {distraction_episodes} interruption(s), but there is room to make the session steadier.",
            "<li>Set one small, specific goal for the next 20–25 minutes.</li>"
            "<li>Place your phone and other likely distractions out of reach before starting.</li>",
        )

    return (
        f"This session had {distraction_episodes} longer interruption(s). A simpler, shorter focus block may help next time.",
        "<li>Try a 15–20 minute focus block followed by a planned short break.</li>"
        "<li>Prepare water, notes, and materials beforehand so you do not need to leave your study space.</li>"
        "<li>Silence notifications before starting the next block.</li>",
    )
