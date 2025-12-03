import argparse
import json
import sys
from type1 import evaluate_answer, StrategySolver, DOMAIN_KEYWORDS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="smartest_type1_questions.json", help="Fisier cu intrebari public (contine params)")
    ap.add_argument("-i", "--index", type=int, default=0, help="Index intrebare din set (0-based)")
    ap.add_argument("-t", "--text", type=str, default=None, help="Raspunsul liber al utilizatorului (string)")
    args = ap.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not (0 <= args.index < len(questions)):
        print(f"Index invalid. Furnizati un index intre 0 si {len(questions)-1}.")
        sys.exit(1)

    qitem = questions[args.index]
    # Recalculam raspunsul corect
    correct_answer, explanation = StrategySolver.solve(qitem.get("problem_type", ""), qitem.get("params", {}))
    # Construim itemul pentru evaluare
    eval_item = {
        "question": qitem.get("question", ""),
        "options": qitem.get("options", []),
        "correct_answer": correct_answer,
        "explanation": explanation,
        "keywords": DOMAIN_KEYWORDS.get(qitem.get("problem_type", ""), [])
    }

    if not args.text:
        print(f"Întrebare: {qitem.get('question')}")
        print("Scrie raspunsul si apasa ENTER (o singura linie):")
        args.text = sys.stdin.readline().strip()

    score = evaluate_answer(args.text, eval_item)
    print("------ Rezultat ------")
    print(f"Întrebare: {qitem.get('question')}")
    print(f"Raspuns utilizator: {args.text}")
    print(f"Scor: {score['score']}")
    print(f"Detalii: {score['components']}")

if __name__ == "__main__":
    main()
