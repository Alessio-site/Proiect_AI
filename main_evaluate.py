import argparse
import json
import sys
from type1 import evaluate_answer, StrategySolver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="smartest_type1_questions.json", 
                    help="Fisier cu intrebari public (contine params)")
    ap.add_argument("-i", "--index", type=int, default=0, 
                    help="Index intrebare din set (0-based)")
    ap.add_argument("-t", "--text", type=str, default=None, 
                    help="Raspunsul liber al utilizatorului (string)")
    args = ap.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    if not (0 <= args.index < len(questions)):
        print(f"Index invalid.  Furnizati un index intre 0 si {len(questions)-1}.")
        sys. exit(1)

    qitem = questions[args. index]
    
    # FIX: solve() returnează DICT, nu tuple! 
    solution = StrategySolver. solve(
        qitem. get("problem_type", ""), 
        qitem.get("params", {})
    )
    correct_answer = solution["best_strategy"]
    explanation = solution["reasoning_summary"]
    calculations = solution. get("calculations", {})

    # Construim itemul pentru evaluare
    eval_item = {
        "question": qitem. get("question", ""),
        "options": qitem.get("options", []),
        "problem_type": qitem.get("problem_type", ""),
        "params": qitem.get("params", {})
    }

    if not args.text:
        print(f"\n{'='*60}")
        print(f"ÎNTREBARE #{args.index}:")
        print(f"{'='*60}")
        print(f"{qitem. get('question')}")
        print(f"\nOPȚIUNI:")
        for i, opt in enumerate(qitem.get('options', []), 1):
            print(f"  {i}.  {opt}")
        print(f"\n{'='*60}")
        print("Scrie răspunsul tău și apasă ENTER:")
        args.text = sys.stdin.readline(). strip()

    # Evaluăm răspunsul
    result = evaluate_answer(args.text, eval_item)
    
    print(f"\n{'='*60}")
    print("REZULTAT EVALUARE")
    print(f"{'='*60}")
    print(f"Răspuns utilizator: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"\n✅ Răspuns corect: {correct_answer}")
    print(f"\n📊 SCOR: {result['score']}/100")
    print(f"\n📋 COMPONENTE:")
    print(f"   • Selecție strategie: {result['components']['selection_score']*100:.0f}%")
    print(f"   • Calitate argumentare: {result['components']['reasoning_score']*100:.0f}%")
    print(f"   • Strategie detectată: {result['components']['detected_intent']}")
    print(f"\n💬 FEEDBACK: {result['components']['message']}")
    
    print(f"\n{'='*60}")
    print("EXPLICAȚIE COMPLETĂ:")
    print(f"{'='*60}")
    print(explanation)

if __name__ == "__main__":
    main()