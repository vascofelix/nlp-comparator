import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from nlp import run_comparison

app = Flask(__name__)

DEFAULT_DOCS_DIR = Path(__file__).parent / "default_docs"
DEFAULT_DOCS_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/default-docs")
def list_default_docs():
    files = sorted(f.name for f in DEFAULT_DOCS_DIR.glob("*.txt"))
    return jsonify(files)


@app.route("/api/default-docs/<name>")
def get_default_doc(name):
    path = DEFAULT_DOCS_DIR / name
    if not path.is_file() or not path.suffix == ".txt":
        return jsonify({"error": "not found"}), 404
    return path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/compare", methods=["POST"])
def compare():
    data = request.get_json()
    url_a = data.get("url_a", "").strip()
    url_b = data.get("url_b", "").strip()
    runs = min(int(data.get("runs", 1)), 10)
    files = data.get("files", [])

    if not url_a or not url_b:
        return jsonify({"error": "Both URLs are required"}), 400
    if not files:
        return jsonify({"error": "At least one file is required"}), 400

    results = []
    for f in files:
        name = f.get("name", "unknown")
        text = f.get("text", "")
        if not text:
            continue
        result = run_comparison(url_a, url_b, text, runs)
        result["file"] = name
        result["chars"] = len(text)
        results.append(result)

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
