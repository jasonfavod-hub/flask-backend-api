from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import json
import tempfile
import subprocess

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

# ---------- Gemini setup ----------
# حاول قراءة مفتاح API من Environment Variable أولاً
GEMINI_API_KEY = os.environ.get("AIzaSyB7nN_7JLZeB5Fmqgqe3qfZWRyJbb3_sAU", "")
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    genai_client = genai
except Exception:
    genai_client = None

# ---------- Utility: run code safely ----------
def execute_code(code: str, language: str, timeout: int = 5):
    start = time.time()
    with tempfile.NamedTemporaryFile(mode='w', suffix=('.py' if language == 'python' else '.js'), delete=False) as f:
        f.write(code)
        fname = f.name

    if language == 'python':
        cmd = ["python3", fname]
    elif language in ('javascript', 'node'):
        cmd = ["node", fname]
    else:
        return {"stdout":"", "stderr": f"Unsupported language: {language}", "returncode": -1, "timed_out": False, "runtime_ms": 0}

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as te:
        stdout = te.stdout.decode() if te.stdout else ""
        stderr = (te.stderr.decode() if te.stderr else "") + f"\nExecution timed out after {timeout} seconds."
        returncode = -1
        timed_out = True
    except Exception as e:
        stdout = ""
        stderr = f"Execution failed: {str(e)}"
        returncode = -1
        timed_out = False
    finally:
        try:
            os.remove(fname)
        except:
            pass

    runtime_ms = int((time.time() - start) * 1000)
    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "timed_out": timed_out, "runtime_ms": runtime_ms}

# ---------- Helper: call Gemini ----------
def call_gemini(prompt: str, model: str = "gemini-2.5-flash"):
    if genai_client is None:
        return {"error": "Gemini client not configured on server."}
    try:
        res = genai_client.generate_text(model=model, prompt=prompt)
        text = getattr(res, "result", None) or getattr(res, "text", None) or str(res)
        return {"text": text}
    except Exception as e:
        return {"error": f"Gemini call failed: {e}"}

# ---------- API Routes ----------
@app.route("/execute", methods=["POST"])
def api_execute():
    data = request.get_json(force=True)
    code = data.get("code", "")
    language = (data.get("language") or "python").lower()
    try:
        timeout = int(data.get("timeout", 5))
    except:
        timeout = 5
    if timeout > 10:
        timeout = 10
    return jsonify(execute_code(code, language, timeout))

@app.route("/ai-check", methods=["POST"])
def api_ai_check():
    data = request.get_json(force=True)
    code = data.get("code", "")
    language = (data.get("language") or "python").lower()
    if not code:
        return jsonify({"error": "No code provided"}), 400

    prompt = (
        "You are a professional programming assistant. Analyze the following code for bugs and style issues. "
        "Provide corrected code and explanation in Arabic.\n\n"
        f"{code}"
    )
    return jsonify(call_gemini(prompt))

@app.route("/ai-generate", methods=["POST"])
def api_ai_generate():
    data = request.get_json(force=True)
    prompt_user = data.get("prompt", "").strip()
    language = (data.get("language") or "python").lower()
    if not prompt_user:
        return jsonify({"error":"No prompt provided"}), 400

    prompt = f"You are a helpful programming assistant. Generate runnable {language} code for:\n{prompt_user}"
    return jsonify(call_gemini(prompt))

@app.route("/ai-explain", methods=["POST"])
def api_ai_explain():
    data = request.get_json(force=True)
    code = data.get("code", "")
    language = (data.get("language") or "python").lower()
    if not code:
        return jsonify({"error": "No code provided"}), 400

    prompt = f"Explain this {language} code in Arabic:\n{code}"
    return jsonify(call_gemini(prompt))

# ---------- Health check ----------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"ok": True, "gemini_configured": bool(GEMINI_API_KEY)})

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
