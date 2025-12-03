# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from type1 import (
    generate_n,
    evaluate_answer,
    save_public_questions,
    save_answer_key,
    StrategySolver # Importăm solverul pentru a afișa explicațiile și răspunsul sistemului
)
import os, secrets, json, datetime

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
REVEAL_AFTER_EVAL = True  # Setat pe True ca să vezi logica dinamică în acțiune
PUBLIC_FILE = "smartest_type1_questions.json"
KEY_FILE = "smartest_type1_answer_key.json"
RESULTS_LOG = "smartest_results.json"

# ------------------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html")

# ------------------------------------------------------------------------------------
@app.post("/generate")
def generate():
    """
    Generează întrebări pe baza numărului și a capitolelor selectate, 
    le salvează pe disc și reîncarcă lista în UI.
    """
    n = int(request.form.get("n", 5))
    
    # NOU: Preluăm lista de capitole selectate din formular (checkbox-uri)
    # Dacă utilizatorul nu selectează nimic, lista va fi goală și generate_n va folosi toate tipurile.
    selected_topics = request.form.getlist("topics")
    
    # Apelăm funcția de generare cu filtrul de topicuri
    items = generate_n(n, selected_topics)

    # Salvăm fișierele. Ambele fișiere conțin parametrii problemei,
    # iar răspunsul corect este calculat dinamic la evaluare (garantând corectitudinea).
    save_public_questions(items, PUBLIC_FILE)
    save_answer_key(items, KEY_FILE)

    session["count"] = len(items)
    return redirect(url_for("questions"))

# ------------------------------------------------------------------------------------
@app.get("/questions")
def questions():
    """Afișează lista publică de întrebări."""
    if not os.path.exists(PUBLIC_FILE):
        return redirect(url_for("home"))

    with open(PUBLIC_FILE, "r", encoding="utf-8") as f:
        public = json.load(f)

    items = [{"index": i, **it} for i, it in enumerate(public)]
    return render_template("questions.html", items=items)

# ------------------------------------------------------------------------------------
@app.get("/q/<int:idx>")
def question(idx: int):
    """Afișează o întrebare + hint (opțiuni)."""
    if not (os.path.exists(PUBLIC_FILE) and os.path.exists(KEY_FILE)):
        return redirect(url_for("home"))

    with open(PUBLIC_FILE, "r", encoding="utf-8") as f:
        public = json.load(f)
    if not (0 <= idx < len(public)):
        return redirect(url_for("questions"))

    it_pub = public[idx]
    # Options sunt generate deja în generate_n și salvate în json
    options = it_pub.get("options", [])

    return render_template(
        "question.html",
        idx=idx,
        total=len(public),
        question=it_pub["question"],
        options=options,
    )

# ------------------------------------------------------------------------------------
@app.post("/evaluate")
def evaluate():
    """Evaluează răspunsul utilizatorului calculând adevărul pe loc."""
    if not os.path.exists(KEY_FILE):
        return redirect(url_for("home"))

    idx = int(request.form.get("idx", 0))
    user_text = (request.form.get("answer", "") or "").strip()

    with open(KEY_FILE, "r", encoding="utf-8") as f:
        key = json.load(f)
    if not (0 <= idx < len(key)):
        return redirect(url_for("questions"))

    item = key[idx] 
    # item conține 'problem_type' și 'params'. 

    # 1. Evaluare Dinamică a Răspunsului Utilizatorului (Scor 0-100%)
    result = evaluate_answer(user_text, item)

    # 2. Generarea Răspunsului de către Aplicație (System Answer)
    # Aceasta îndeplinește cerința: "Aplicația este capabilă să răspundă la întrebări... punctual sau detaliat"
    system_response = StrategySolver.get_system_answer(item["problem_type"], item["params"])

    # 3. Pregătire date pentru UI (pentru compatibilitate cu result.html vechi sau nou)
    # Păstrăm și structura veche 'reveal' dacă template-ul o folosește încă, 
    # dar 'system_response' este varianta robustă.
    reveal = None
    if REVEAL_AFTER_EVAL:
        reveal = {
            "correct_answer": system_response["punctual"],
            "explanation": system_response["detailed"], # Putem folosi detaliul aici
        }

    # Baseline (scor parțial simplu, păstrat doar pentru consistență vizuală)
    partial_baseline = result["score"]

    # -------------------- LOG pe disc --------------------
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index": idx,
        "question": item["question"],
        "problem_type": item.get("problem_type"),
        "params": item.get("params"),
        "user_answer": user_text,
        "score": result["score"],
        "components": result["components"],
        "calculated_truth": system_response["punctual"]
    }
    
    try:
        data = []
        if os.path.exists(RESULTS_LOG):
            with open(RESULTS_LOG, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append(log_entry)
        with open(RESULTS_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Nu pot scrie {RESULTS_LOG}: {e}")

    return render_template(
        "result.html",
        idx=idx,
        total=len(key),
        user_text=user_text,
        score=result,
        baseline=partial_baseline,
        reveal=reveal,
        system_response=system_response, # Trimitem obiectul complet către template
        next_idx=idx + 1 if idx + 1 < len(key) else None,
        prev_idx=idx - 1 if idx - 1 >= 0 else None,
    )

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)