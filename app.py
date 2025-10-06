from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import subprocess
import tempfile

# إعداد مفتاح Gemini API
genai.configure(api_key="ضع_مفتاحك_هنا")

# إنشاء التطبيق
app = Flask(__name__)
CORS(app)


# main.py
"""
Flask backend for Programming Assistant — Code execution and Gemini-powered AI.
Intended for deployment on a free Python host (e.g., Replit). Configure GEMINI_API_KEY
and ensure the environment has `node` available for executing JavaScript.

Endpoints:
- POST /execute   -> { code, language, timeout }   : runs code (python/node), returns stdout/stderr/json
- POST /ai-check  -> { code, language }             : asks Gemini to check/fix code, returns ai_text
- POST /ai-generate -> { prompt, language }         : asks Gemini to generate code per prompt
- POST /ai-explain -> { code, language }           : asks Gemini to explain given code (Arabic)
"""

import os
import time
import json
import tempfile
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- Gemini (Gemini / Google Generative AI) setup ----------
# This backend expects you to set an environment variable named GEMINI_API_KEY
# or configure Google client credentials. The code below attempts to use the
# official-looking SDK surface. Depending on your installed package, adjust imports.
#
# Recommended: set GEMINI_API_KEY in the Replit/environ settings.
GEMINI_API_KEY = os.environ.get("AIzaSyB7nN_7JLZeB5Fmqgqe3qfZWRyJbb3_sAU", "")  # <<-- set this in your deploy environment

# We'll attempt to import the "google" genai client as used by official docs.
# There are several older/newer python packages; this tries to support common cases.
genai_client = None
genai_lib = None
try:
    # preferred new-ish import style
    from google import genai
    genai_client = genai.Client()  # this client will read env creds
    genai_lib = 'google.genai'
except Exception as e:
    try:
        import google.generativeai as genai_old
        # the older package often uses configure()
        if GEMINI_API_KEY:
            genai_old.configure(api_key=GEMINI_API_KEY)
        genai_client = genai_old
        genai_lib = 'google.generativeai'
    except Exception as e2:
        # If neither import works, we'll still start the server but AI endpoints will return errors.
        genai_client = None
        genai_lib = None

# If an API key exists, try to set it into env (some SDKs pick this up automatically)
if GEMINI_API_KEY:
    os.environ.setdefault("GENAI_API_KEY", GEMINI_API_KEY)
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Utility: run code safely (best-effort sandboxing)
def execute_code(code: str, language: str, timeout: int = 5):
    """
    Execute code in a temporary file using subprocess.
    Supports 'python' and 'javascript' (node).
    Returns dict: stdout, stderr, returncode, timed_out, runtime_ms
    """
    start = time.time()
    with tempfile.NamedTemporaryFile(mode='w', suffix=('.py' if language == 'python' else '.js'), delete=False) as f:
        f.write(code)
        fname = f.name

    if language == 'python':
        cmd = ["python3", fname]
    elif language == 'javascript' or language == 'node':
        cmd = ["node", fname]
    else:
        return {"stdout":"", "stderr": f"Unsupported language: {language}", "returncode": -1, "timed_out": False, "runtime_ms": 0}

    # Run subprocess with timeout
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        timed_out = False
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as te:
        stdout = te.stdout.decode() if te.stdout else ""
        stderr = (te.stderr.decode() if te.stderr else "") + f"\nExecution timed out after {timeout} seconds."
        timed_out = True
        returncode = -1
    except Exception as e:
        stdout = ""
        stderr = f"Execution failed: {str(e)}"
        timed_out = False
        returncode = -1
    finally:
        try:
            os.remove(fname)
        except:
            pass

    runtime_ms = int((time.time() - start) * 1000)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_ms": runtime_ms
    }

# ---------- Helper: call Gemini / GenAI ----------
def call_gemini(prompt: str, system: str = None, model: str = "gemini-2.5-flash", **kwargs):
    """
    Call the Gemini / Google GenAI client.
    This function attempts to work with multiple client library shapes.
    It returns the text returned by the model (string).
    """
    if genai_client is None:
        return {"error": "Gemini client not configured on server. Please set GEMINI_API_KEY and install the appropriate SDK."}

    try:
        # Preferred: google.genai Client usage (cloud docs)
        if genai_lib == 'google.genai':
            # The client object exposes client.models.generate_content(...)
            request_contents = prompt
            # If system instruction provided, include in config if supported
            response = genai_client.models.generate_content(model=model, contents=request_contents)
            # Try to extract textual result
            text = getattr(response, "text", None)
            if not text:
                # Some responses embed the output in a nested structure
                try:
                    # examine response to find textual content
                    text = json.dumps(response, default=str)
                except:
                    text = str(response)
            return {"text": text}
        elif genai_lib == 'google.generativeai':
            # older google.generativeai package (genai_old.generate_text or genai_old.chat)
            # Use generate_text if available
            try:
                res = genai_client.generate_text(model=model, prompt=prompt)
                # res may have .result or .text
                text = getattr(res, "result", None) or getattr(res, "text", None) or str(res)
                return {"text": text}
            except Exception:
                # chat style
                try:
                    res = genai_client.chat(model=model, messages=[{"role":"user","content":prompt}])
                    # Extract message(s)
                    text = res.last or getattr(res, "result", None) or str(res)
                    return {"text": text}
                except Exception as ex:
                    return {"error": f"Gemini call failed: {ex}"}
        else:
            # Last resort: try to call method generically
            try:
                response = genai_client.models.generate_content(model=model, contents=prompt)
                text = getattr(response, "text", None) or str(response)
                return {"text": text}
            except Exception as e:
                return {"error": f"Unable to call Gemini client: {e}"}
    except Exception as e:
        return {"error": f"Gemini invocation error: {e}"}

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
    # Enforce a hard cap (safety)
    if timeout > 10:
        timeout = 10

    result = execute_code(code, language, timeout)
    return jsonify(result)

@app.route("/ai-check", methods=["POST"])
def api_ai_check():
    """
    Expect JSON: { code, language }
    Returns: { ai_text: "<full reply from Gemini>" }
    The prompt instructs Gemini to analyze for bugs, return corrected code and a detailed explanation in Arabic.
    """
    data = request.get_json(force=True)
    code = data.get("code", "")
    language = (data.get("language") or "python").lower()

    if not code:
        return jsonify({"error": "No code provided"}), 400

    prompt = (
        "You are a professional programming assistant. Analyze the following {lang} source code for bugs, "
        "security issues, and stylistic problems. Provide:\n\n"
        "1) A corrected and runnable version of the code enclosed in a triple-backtick code block with the correct language tag.\n"
        "2) A detailed explanation in Arabic of what was wrong, why the changes were made, and suggestions for improvement/refactoring. "
        "Also include small tips for testing and edge cases.\n\n"
        "Return the corrected code first inside a markdown code fence, then the Arabic explanation. Do not include any text other than the code block and the Arabic explanation.\n\n"
        "Original code:\n\n"
        "```{lang}\n{code}\n```\n"
    ).format(lang=language, code=code)

    ai_resp = call_gemini(prompt)
    if "error" in ai_resp:
        return jsonify({"error": ai_resp["error"]}), 500

    return jsonify({"ai_text": ai_resp.get("text", "")})

@app.route("/ai-generate", methods=["POST"])
def api_ai_generate():
    """
    Expect JSON: { prompt, language }
    Returns: { ai_text: "<assistant response containing code in a code block>" }
    """
    data = request.get_json(force=True)
    prompt_user = data.get("prompt", "").strip()
    language = (data.get("language") or "python").lower()
    if not prompt_user:
        return jsonify({"error":"No prompt provided"}), 400

    prompt = (
        "You are a helpful programming assistant. The user requested the following: \n\n"
        "{user_prompt}\n\n"
        "Please generate complete, runnable {lang} code that fulfills the request. "
        "Return the code inside a triple-backtick fenced code block with the language tag, "
        "and after the code include a short explanation (1-3 sentences). "
        "Keep explanations concise. Include any install/runtime notes if needed.\n\n"
        "IMPORTANT: Format the code so the frontend can show a 'Replace Editor Content' button when it finds a code fence.\n"
    ).format(user_prompt=prompt_user, lang=language)

    ai_resp = call_gemini(prompt)
    if "error" in ai_resp:
        return jsonify({"error": ai_resp["error"]}), 500

    return jsonify({"ai_text": ai_resp.get("text", "")})

@app.route("/ai-explain", methods=["POST"])
def api_ai_explain():
    """
    Expect JSON: { code, language }
    Returns: { ai_text: "<explanation in Arabic>" }
    """
    data = request.get_json(force=True)
    code = data.get("code", "")
    language = (data.get("language") or "python").lower()

    if not code:
        return jsonify({"error": "No code provided"}), 400

    prompt = (
        "You are an expert programmer and teacher. Explain the following {lang} code in Arabic: what it does, "
        "how the main functions work, important variables, and potential pitfalls. Keep explanation clear and suitable "
        "for a developer who knows the basics but needs a thorough walkthrough. Use Arabic.\n\n"
        "Code:\n```{lang}\n{code}\n```"
    ).format(lang=language, code=code)

    ai_resp = call_gemini(prompt)
    if "error" in ai_resp:
        return jsonify({"error": ai_resp["error"]}), 500

    return jsonify({"ai_text": ai_resp.get("text", "")})

# ---------- Simple health check ----------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"ok": True, "gemini_lib": genai_lib or "none", "gemini_configured": bool(GEMINI_API_KEY)})

# ---------- Run (for local testing) ----------
if __name__ == "__main__":
    # Warning: Flask debug mode is convenient for development but not for production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
