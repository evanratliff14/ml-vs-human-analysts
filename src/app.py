import Flask, jsonify, send_file, request
import pandas as pd
from pathlib import Path
p = Path(__file__).parent / "data" / "players.csv"

p.exists()        # True if file exists
p.is_file()       # True if it's a file
p.read_text()     # read whole file (string)
p.open("rb")      # open as binary file-like object
p.resolve()       # absolute canonical path
p.parent          # parent dir
list(p.glob("*.csv"))   # non-recursive
for f in p.rglob("*.parquet"):   # recursive

class App:
    def __init__:
        app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"

# @app.route(...) is a decorator that registers the function below it as an HTTP handler for that path.

# The decorated function becomes the view or endpoint handler; its return value is converted into an HTTP response by Flask.

# Common return types:

# dict (Flask converts to JSON) — return {"ok": True}, 200

# str (plain text/html)

# (body, status, headers) tuple

# send_file(...) for files/streams

# methods controls allowed HTTP methods. Default is ["GET"].

# Path variables (like <name>) are passed as function args. Add converters: <int:id>, <path:subpath>.

# Use @app.route for simple apps. For larger apps use Blueprints, same decorator style: @bp.route(...).
    @app.route()
    def 
    
