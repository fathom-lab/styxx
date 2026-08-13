// The credit tripwire from the rented-stack audit (tier 0.3, ranked fix #3).
// Watches the gateway log for a credit/quota refusal and cuts darkflobi over to
// his own silicon. One-way by design: cutting BACK when credits return is an
// operator decision, not a daemon's.
module.exports = {
  apps: [
    {
      name: "brain-failover",
      script: "C:\\Users\\heyzo\\.styxx\\glimmer-day-zero\\brain_failover.py",
      interpreter: "C:\\Users\\heyzo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      cwd: "C:\\Users\\heyzo\\.styxx\\glimmer-day-zero",
      autorestart: true,
      max_restarts: 20,
      windowsHide: true,
      env: { PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
    },
  ],
};
