#!/usr/bin/env python3
"""
STT Application Health Monitor
Production-ready monitoring and health checks.
"""
import time
import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path

class STTHealthMonitor:
    def __init__(self):
        self.monitoring = True
        self.check_interval = 300  # 5 minutes

    def run_health_checks(self):
        timestamp = datetime.now().isoformat()
        report = {
            "timestamp": timestamp,
            "status": "healthy",
            "checks": {}
        }
        # Check stt command
        try:
            result = subprocess.run(
                [sys.executable, "-m", "stt", "--version"],
                capture_output=True, text=True, timeout=10
            )
            report["checks"]["stt_command"] = {
                "status": "available" if result.returncode == 0 else "unavailable",
                "details": result.stdout.strip() if result.returncode == 0 else result.stderr
            }
        except Exception as e:
            report["checks"]["stt_command"] = {"status": "error", "details": str(e)}

        # Check config
        try:
            from stt import config
            cfg = config.load()
            report["checks"]["configuration"] = {
                "status": "valid",
                "models_dir": cfg.get("models_dir"),
                "threads": cfg.get("threads")
            }
        except Exception as e:
            report["checks"]["configuration"] = {"status": "invalid", "details": str(e)}

        # Determine overall health
        critical_failures = sum(
            1 for r in report["checks"].values()
            if r.get("status") in ["error", "invalid", "unavailable"]
        )
        report["status"] = "healthy" if critical_failures == 0 else "degraded"
        return report

    def start_monitoring(self):
        print("Starting STT application monitoring (Ctrl+C to stop)...")
        while self.monitoring:
            try:
                report = self.run_health_checks()
                log_file = Path("health.log")
                with open(log_file, "a") as f:
                    f.write(json.dumps(report) + "\n")
                status_icon = {"healthy": "OK", "degraded": "WARN", "monitoring": "INFO"}[report["status"]]
                print(f"[{status_icon}] {report['timestamp']} - status: {report['status']}")
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    STTHealthMonitor().start_monitoring()
