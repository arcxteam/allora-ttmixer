# Gunicorn config variables
from config import FLASK_PORT

loglevel = "info"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 30
keepalive = 5
worker_class = "gthread"
workers = 1
threads = 8
bind = f"0.0.0.0:{FLASK_PORT}"
