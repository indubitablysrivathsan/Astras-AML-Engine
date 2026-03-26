"""
Audit Trail Service
Immutable audit trail management for SAR generation process.
Every decision is fully auditable and regulator-ready.
"""
import json
import os
import sqlite3
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, OUTPUTS_DIR


def create_audit_tables(db_path=DB_PATH):
    """Create audit trail tables in the database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT,
            user_id TEXT DEFAULT 'system',
            FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sar_records (
            sar_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            narrative TEXT NOT NULL,
            risk_score REAL,
            bsi_score REAL,
            typology TEXT,
            compliance_status TEXT,
            status TEXT DEFAULT 'draft',
            generated_at TEXT NOT NULL,
            reviewed_by TEXT,
            approved_by TEXT,
            filed_date TEXT,
            audit_json TEXT,
            FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
        )
    """)
    conn.commit()
    conn.close()


def log_action(alert_id, customer_id, action, details=None, user_id='system', db_path=DB_PATH):
    """Log an action to the audit trail."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO audit_trail (alert_id, customer_id, action, timestamp, details, user_id) VALUES (?,?,?,?,?,?)",
        (alert_id, customer_id, action, datetime.now().isoformat(),
         json.dumps(details) if details else None, user_id)
    )
    conn.commit()
    conn.close()


def save_sar_record(alert_id, customer_id, narrative, audit_trail, compliance_check, db_path=DB_PATH):
    """Save a SAR record to the database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO sar_records
        (alert_id, customer_id, narrative, risk_score, bsi_score, typology,
         compliance_status, status, generated_at, audit_json)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        alert_id,
        customer_id,
        narrative,
        audit_trail.get('risk_score'),
        audit_trail.get('bsi_score'),
        audit_trail.get('typology_detected'),
        'compliant' if compliance_check.get('compliant') else 'review_required',
        'draft',
        datetime.now().isoformat(),
        json.dumps(audit_trail, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()

    log_action(alert_id, customer_id, 'sar_generated', {
        'risk_score': audit_trail.get('risk_score'),
        'word_count': audit_trail.get('narrative_word_count'),
        'compliant': compliance_check.get('compliant'),
    })


def get_audit_history(alert_id, db_path=DB_PATH):
    """Get full audit history for an alert."""
    conn = sqlite3.connect(db_path)
    history = []
    try:
        cursor = conn.execute(
            "SELECT * FROM audit_trail WHERE alert_id = ? ORDER BY timestamp",
            (alert_id,)
        )
        cols = [d[0] for d in cursor.description]
        for row in cursor.fetchall():
            entry = dict(zip(cols, row))
            if entry.get('details'):
                entry['details'] = json.loads(entry['details'])
            history.append(entry)
    except Exception:
        pass
    conn.close()
    return history


def get_sar_record(alert_id, db_path=DB_PATH):
    """Get the latest SAR record for an alert."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM sar_records WHERE alert_id = ? ORDER BY generated_at DESC LIMIT 1",
            (alert_id,)
        )
        cols = [d[0] for d in cursor.description]
        row = cursor.fetchone()
        if row:
            record = dict(zip(cols, row))
            if record.get('audit_json'):
                record['audit_json'] = json.loads(record['audit_json'])
            return record
    except Exception:
        pass
    finally:
        conn.close()
    return None


def load_sar_from_file(alert_id, outputs_dir=OUTPUTS_DIR):
    """Load a SAR document from the outputs directory."""
    filepath = os.path.join(outputs_dir, f'sar_alert_{alert_id}.json')
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_generated_sars(outputs_dir=OUTPUTS_DIR):
    """List all generated SARs."""
    import glob
    files = sorted(glob.glob(os.path.join(outputs_dir, 'sar_alert_*.json')))
    sars = []
    for f in files:
        alert_num = os.path.basename(f).replace('sar_alert_', '').replace('.json', '')
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                doc = json.load(fh)
                audit = doc.get('audit_trail', {})
                sars.append({
                    'alert_id': int(alert_num),
                    'customer_name': audit.get('customer_name', 'Unknown'),
                    'risk_score': audit.get('risk_score', 0),
                    'typology': audit.get('typology_detected', 'Unknown'),
                    'status': audit.get('status', 'draft'),
                    'generated_at': doc.get('generated_at', ''),
                    'filepath': f,
                })
        except Exception:
            pass
    return sars
