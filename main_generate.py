import argparse
from type1 import generate_n, save_public_questions, save_answer_key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=5, help="Numarul de intrebari de generat")
    ap.add_argument("--public", default="smartest_type1_questions.json", help="Fisier output public (fara raspunsuri)")
    ap.add_argument("--key", default="smartest_type1_answer_key.json", help="Fisier output answer key (privat)")
    args = ap.parse_args()

    items = generate_n(args.num)
    save_public_questions(items, args.public)
    save_answer_key(items, args.key)
    print(f"✔ Generat {args.num} intrebari.")
    print(f"   Public: {args.public}")
    print(f"   Answer key (privat): {args.key}")

if __name__ == "__main__":
    main()
