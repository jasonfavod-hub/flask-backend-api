from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import json
import tempfile
import subprocess

# ---------- Flask app ----------
app = Flask(__name__)
# تمكين CORS لجميع الأصول، لضمان اتصال الفرونت إند بالباك إند
CORS(app)

# ---------- Gemini setup ----------
# قراءة مفتاح API من Environment Variable مباشرة (GEMINI_API_KEY)
# يجب أن تضع هذا المفتاح في إعدادات Render (Service Settings -> Environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    genai_client = genai
except Exception:
    # سيتم تشغيل هذا إذا فشل استيراد الحزمة أو التهيئة
    genai_client = None

# ---------- Utility: run code safely ----------
def execute_code(code: str, language: str, timeout: int = 5):
    """ينفذ الكود البرمجي في بيئة معزولة ومؤقتة."""
    start = time.time()
    # استخدام اسم ملف مؤقت لكتابة الكود
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
        # تنفيذ الكود مع تحديد مهلة (Timeout)
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
        stderr = f"Execution failed: {str(e)} (Check if 'python3' or 'node' are available on the server)"
        returncode = -1
        timed_out = False
    finally:
        # حذف الملف المؤقت بعد التنفيذ
        try:
            os.remove(fname)
        except:
            pass

    runtime_ms = int((time.time() - start) * 1000)
    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "timed_out": timed_out, "runtime_ms": runtime_ms}

# ---------- Helper: call Gemini ----------
def call_gemini(prompt: str, model: str = "gemini-2.5-flash"):
    """يتصل بواجهة برمجة تطبيقات Gemini لتوليد النصوص."""
    if genai_client is None:
        return {"error": "Gemini client not configured on server. Check API Key."}
    try:
        # ملاحظة: تم تعديل طريقة الاتصال لاستخدام generate_content بدلاً من generate_text
        # لضمان التوافق مع أحدث إصدارات مكتبة Google GenAI
        res = genai_client.models.generate_content(model=model, contents=prompt)
        text = res.text
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
        f"```{language}\n{code}\n```"
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
