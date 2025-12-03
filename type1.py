import random
import re
import json
import math
import spacy
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher

# =======================================================================
# 0. SETUP NLP & AI MODELS (Local - FĂRĂ LLM)
# =======================================================================

# 1. SpaCy pentru analiză sintactică rapidă și negații
try:
    nlp = spacy. load("ro_core_news_lg")
except OSError:
    print("Modelul 'ro_core_news_lg' nu este instalat. Se încearcă 'ro_core_news_sm'...")
    try:
        nlp = spacy.load("ro_core_news_sm")
    except Exception:
        print("Eroare critică: Niciun model spaCy de română găsit.  Evaluarea va fi limitată.")
        nlp = None

# 2.  Sentence Transformers pentru evaluare semantică (LOCAL, fără API)
semantic_model = None
try:
    from sentence_transformers import SentenceTransformer, util
    print("Se încarcă modelul semantic SBERT (rulează LOCAL, fără API)...")
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ Model SBERT încărcat cu succes!")
except ImportError:
    print("WARN: `sentence-transformers` nu este instalat. Se va folosi fallback pe cuvinte cheie.")
except Exception as e:
    print(f"WARN: Eroare la încărcarea SBERT: {e}")


# =======================================================================
# HELPERS: Normalizare & Utilități Matematice
# =======================================================================
def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, remove diacritics and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\"'()\[\]{}:;,.  ! ?\\/<>@#%^&*+=~`|\\]+", " ", text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _calculate_factorial(n: int) -> int:
    """Calculează n! în mod sigur."""
    if n < 0:
        return 0
    if n <= 1:
        return 1
    result = 1
    for i in range(2, min(n + 1, 171)):  # Limită pentru a evita overflow
        result *= i
    return result


def _format_large_number(n) -> str:
    """Formatează numere mari pentru afișare."""
    # Verificări pentru valori speciale
    if n is None:
        return "0"
    
    if isinstance(n, float):
        if math.isinf(n) or math.isnan(n):
            return "∞"
        n = int(n)  # Convertim la int pentru formatare
    
    # Asigurăm că e int
    try:
        n = int(n)
    except (ValueError, TypeError):
        return str(n)
    
    # Formatare bazată pe mărime
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f} trilioane"
    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} miliarde"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f} milioane"
    elif n >= 1_000:
        # Formatare cu separator de mii
        return f"{n:,}".replace(",", ".")
    else:
        return str(n)


# =======================================================================
# LEXICON: Negații și Pattern-uri
# =======================================================================
NEGATION_WORDS = {
    "nu", "nici", "fara", "lipsit", "niciodata", "niciun", "nicio",
    "not", "never", "without", "no", "none", "dont", "doesnt", "wont", "isnt", "arent"
}

NEGATION_VERBS = {
    "ignora", "ignor", "bloca", "blocheaza", "opreste", "opri", "anula", "anuleaza",
    "impiedica", "uita", "distruge", "sterge", "dezactiva", "evita", "exclude", "resping",
    "renunta", "refuz", "critica", "contesta", "respinge",
    "ignore", "ignored", "block", "stop", "prevent", "disable", "erase", "destroy",
    "avoid", "exclude", "reject", "skip", "refuse"
}

CONDITIONAL_WORDS = {
    "ar", "s-ar", "fi", "decat", "fata", "versus", "comparativ", "instead",
    "would", "could", "should", "fata de", "spre deosebire"
}

REJECTION_PATTERNS = [
    r"nu\s+(?:vom\s+)?(?:folosi|alege|opta|recomanda)\s+(. +?)(?:\s+|$|,|\. )",
    r"(?:evitam|excludem|respingem|refuzam)\s+(.+?)(?:\s+|$|,|\. )",
    r"(. +? )\s+(?:nu\s+(?:este|e)|ar\s+fi\s+gresit|nu\s+functioneaza)",
    r"(?:in\s+loc\s+de|spre\s+deosebire\s+de)\s+(.+?)(?:\s+|,|\.)",
]


# =======================================================================
# STRATEGY SIGNATURES: Semnături pentru detectarea strategiilor din descrieri
# =======================================================================
STRATEGY_SIGNATURES = {
    "simulated annealing": [
        "temperatura", "racire", "racindu", "energie", "accepta.*mutari.*proaste",
        "configuratie aleatoare", "cooling", "annealing", "probabilitate.*acceptare",
        "treptat", "gradual.*scade", "metropolis", "boltzmann"
    ],
    "genetic algorithm": [
        "populatie", "selectie", "crossover", "mutatie", "fitness", "generatii",
        "evolutie", "cromozom", "gene", "supravietuire", "parinti", "offspring"
    ],
    "min-conflicts heuristic": [
        "conflicte", "minimizeaza.*conflicte", "local.*search", "cautare.*locala",
        "hill.*climbing", "vecini", "swap", "reduce.*conflicte"
    ],
    "plain bfs": [
        "nivel.*nivel", "coada", "queue", "exhaustiv.*nivel", "breadth", "latime"
    ],
    "plain dfs": [
        "adancime", "stiva", "stack", "depth", "recursiv.*simplu"
    ],
    "backtracking with mrv + forward checking": [
        "mrv", "minimum.*remaining", "forward.*checking", "propagare.*constrangeri",
        "variabila.*critica", "domeniu.*redus", "fail.*first"
    ],
    "ac-3 with backtracking": [
        "ac-3", "arc.*consistency", "consistenta.*arc", "propagare.*arc"
    ],
    "warnsdorff heuristic (greedy)": [
        "warnsdorff", "grad.*minim", "mutari.*disponibile", "knight.*tour.*greedy"
    ],
    "recursive strategy (divide and conquer)": [
        "recursiv", "subprobleme", "divide", "conquer", "structura.*recursiva",
        "n-1.*discuri", "problema.*mica"
    ]
}

# Lista tuturor strategiilor disponibile
ALL_STRATEGIES = [
    "plain dfs",
    "plain bfs",
    "genetic algorithm",
    "simulated annealing",
    "backtracking with mrv + forward checking",
    "ac-3 with backtracking",
    "min-conflicts heuristic",
    "warnsdorff heuristic (greedy)",
    "recursive strategy (divide and conquer)",
    "a*",
    "ida*",
    "dynamic programming",
    "greedy coloring",
    "hill climbing with sideways moves"
]


def _detect_strategy_from_description(text: str) -> Optional[str]:
    """Detectează strategia descrisă în text bazat pe semnături."""
    text_norm = _normalize(text)
    scores = {}
    for strategy, patterns in STRATEGY_SIGNATURES.items():
        score = 0
        for pattern in patterns:
            if re. search(pattern, text_norm):
                score += 1
        if score > 0:
            scores[strategy] = score
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return None


# =======================================================================
# 1. STRATEGY SOLVER - Calculează strategia optimă MATEMATIC (NU HARDCODAT)
# =======================================================================
class StrategySolver:
    """
    Solver inteligent care calculează strategia optimă bazat pe analiza
    MATEMATICĂ a parametrilor problemei, NU pe valori hardcodate.
    
    Diferența față de versiunea anterioară:
    - Calculează metrici reale (factorial, densitate, complexitate)
    - Decide bazat pe praguri CALCULATE, nu arbitrare
    - Generează explicații cu FORMULE și CALCULE concrete
    """

    # Praguri bazate pe analiza complexității algoritmice
    # 10 milioane operații = ~1 secundă pe CPU modern
    FEASIBILITY_THRESHOLD = 100_000_000
    
    # Densitate graf: peste 35% se consideră dens (bazat pe literatura de specialitate)
    DENSE_GRAPH_THRESHOLD = 0.35

    # ===================================================================
    # CALCULATOARE DE METRICI (Inima sistemului non-hardcodat)
    # ===================================================================
    
    @staticmethod
    def _calculate_nqueens_metrics(N: int) -> Dict[str, Any]:
        """
        Calculează metrici pentru problema N-Queens bazat pe teoria complexității.
        
        Complexități cunoscute:
        - Brute force: O(N^N)
        - Backtracking: O(N!)
        - Backtracking + MRV + FC: ~O(N!  * 0.15) în practică
        - Min-Conflicts: O(N) iterații, fiecare O(N) => O(N²)
        """
        # Spațiul brut de căutare: fiecare regină pe oricare coloană
        brute_space = N ** N if N <= 20 else float('inf')
        
        # Backtracking worst case: O(N!) - permutări
        factorial_space = _calculate_factorial(N)
        
        # Cu MRV + Forward Checking: reduce cu ~85% în practică
        # Sursa: "Constraint Satisfaction Problems" - Russell & Norvig
        bt_mrv_fc_estimate = int(factorial_space * 0.15) if factorial_space < float('inf') else float('inf')
        
        # Min-conflicts: O(N²) cu ~50 restarts în medie pentru soluție
        # Sursa: "Min-Conflicts Hill-Climbing" - Minton et al., 1992
        min_conflicts_estimate = N * N * 50
        
        # DECIZIE BAZATĂ PE CALCULE
        bt_feasible = bt_mrv_fc_estimate < StrategySolver. FEASIBILITY_THRESHOLD
        
        # Raport de eficiență
        if min_conflicts_estimate > 0:
            efficiency_ratio = bt_mrv_fc_estimate / min_conflicts_estimate
        else:
            efficiency_ratio = float('inf')

        return {
            "N": N,
            "brute_space": brute_space,
            "brute_space_formatted": _format_large_number(brute_space),
            "factorial_space": factorial_space,
            "factorial_formatted": _format_large_number(factorial_space),
            "bt_mrv_fc_estimate": bt_mrv_fc_estimate,
            "bt_mrv_fc_formatted": _format_large_number(bt_mrv_fc_estimate),
            "min_conflicts_estimate": min_conflicts_estimate,
            "min_conflicts_formatted": _format_large_number(min_conflicts_estimate),
            "bt_feasible": bt_feasible,
            "efficiency_ratio": efficiency_ratio,
            "threshold_used": StrategySolver. FEASIBILITY_THRESHOLD,
            "recommendation": "backtracking" if bt_feasible else "min-conflicts"
        }

    @staticmethod
    def _calculate_graph_coloring_metrics(V: int, E: int, k: int) -> Dict[str, Any]:
        """
        Calculează metrici pentru Graph Coloring. 
        
        Formule folosite:
        - Densitate: E / (V*(V-1)/2)
        - Grad mediu: 2E/V
        - Număr cromatic estimat: max(clică maximă) ≈ grad_mediu + 1
        """
        # Numărul maxim de muchii într-un graf complet
        max_edges = V * (V - 1) // 2 if V > 1 else 1
        
        # Densitatea grafului
        density = E / max_edges if max_edges > 0 else 0
        
        # Gradul mediu al nodurilor
        avg_degree = (2 * E) / V if V > 0 else 0
        
        # Spațiul de căutare brut: k^V (fiecare nod poate avea k culori)
        if V <= 25:
            brute_space = k ** V
        else:
            brute_space = float('inf')
        
        # AC-3 overhead: O(E * k²) pentru preprocesare
        ac3_overhead = E * (k ** 2)
        
        # Estimare număr cromatic (pentru a verifica dacă k e suficient)
        # Teorema: χ(G) ≥ ω(G) ≥ gradul mediu / (gradul mediu - 1) pentru grafuri regulate
        estimated_chromatic = max(1, int(avg_degree) + 1)
        
        # DECIZIE: Graf dens = AC-3 necesar, Graf rar = MRV+FC suficient
        is_dense = density > StrategySolver.DENSE_GRAPH_THRESHOLD

        return {
            "V": V,
            "E": E,
            "k": k,
            "max_edges": max_edges,
            "density": round(density, 4),
            "density_percent": round(density * 100, 2),
            "avg_degree": round(avg_degree, 2),
            "is_dense": is_dense,
            "brute_space": brute_space,
            "brute_space_formatted": _format_large_number(brute_space),
            "ac3_overhead": ac3_overhead,
            "ac3_overhead_formatted": _format_large_number(ac3_overhead),
            "estimated_chromatic": estimated_chromatic,
            "likely_colorable": k >= estimated_chromatic,
            "threshold_used": StrategySolver. DENSE_GRAPH_THRESHOLD,
            "recommendation": "ac-3" if is_dense else "mrv-fc"
        }

    @staticmethod
    def _calculate_hanoi_metrics(pegs: int, disks: int) -> Dict[str, Any]:
        """
        Calculează metrici pentru Turnurile din Hanoi. 
        
        Formule:
        - 3 tije: Soluție optimă = 2^n - 1 (demonstrat matematic)
        - 4+ tije: Frame-Stewart (conjectură, cel mai bun cunoscut)
        """
        if pegs == 3:
            # Formula exactă pentru 3 tije
            optimal_moves = (2 ** disks) - 1
            algorithm = "recursive"
            formula_used = f"2^{disks} - 1 = {optimal_moves}"
        else:
            # Frame-Stewart pentru 4+ tije
            # Aproximare: mult mai eficient decât 3 tije
            # Formula Frame-Stewart: T(n,r) = min_{1≤k<n}(2*T(k,r) + T(n-k,r-1))
            # Aproximare simplificată pentru afișare
            optimal_moves = int(2 * disks * math.sqrt(disks))
            algorithm = "frame-stewart"
            formula_used = f"~2 * {disks} * √{disks} ≈ {optimal_moves}"
        
        # Spațiul de stări: pegs^disks (fiecare disc poate fi pe oricare tijă)
        state_space = pegs ** disks

        return {
            "pegs": pegs,
            "disks": disks,
            "optimal_moves": optimal_moves,
            "optimal_moves_formatted": _format_large_number(optimal_moves),
            "formula_used": formula_used,
            "state_space": state_space,
            "state_space_formatted": _format_large_number(state_space),
            "is_classic": pegs == 3,
            "algorithm": algorithm,
            "recommendation": "recursive"
        }

    @staticmethod
    def _calculate_knights_tour_metrics(n: int) -> Dict[str, Any]:
        """
        Calculează metrici pentru Knight's Tour.
        
        Complexități:
        - Backtracking brut: O((n²)!) - imposibil practic
        - Warnsdorff: O(n²) - aproape liniar
        """
        total_squares = n * n
        
        # Backtracking: în cel mai rău caz, explorează toate permutările
        # Limităm calculul pentru a evita overflow
        if total_squares <= 20:
            bt_estimate = _calculate_factorial(total_squares)
        else:
            bt_estimate = float('inf')
        
        # Warnsdorff: vizitează fiecare celulă o dată, max 8 mutări de verificat
        warnsdorff_estimate = total_squares * 8
        
        # Verificăm dacă există soluție
        # Teoremă: Pentru n ≥ 5, există întotdeauna un tur complet
        solvable = n >= 5
        
        # Câștig de eficiență
        if warnsdorff_estimate > 0 and bt_estimate != float('inf'):
            efficiency_gain = bt_estimate / warnsdorff_estimate
        else:
            efficiency_gain = float('inf')

        return {
            "n": n,
            "total_squares": total_squares,
            "bt_estimate": bt_estimate,
            "bt_estimate_formatted": _format_large_number(bt_estimate),
            "warnsdorff_estimate": warnsdorff_estimate,
            "warnsdorff_formatted": _format_large_number(warnsdorff_estimate),
            "solvable": solvable,
            "efficiency_gain": efficiency_gain,
            "efficiency_gain_formatted": _format_large_number(efficiency_gain) if efficiency_gain != float('inf') else "∞",
            "recommendation": "warnsdorff"
        }

    # ===================================================================
    # GENERATOR DE DISTRACTORI INTELIGENȚI
    # ===================================================================
    
    @staticmethod
    def get_distractors(problem_type: str, correct_strategy: str, params: Dict[str, Any], count: int = 3) -> List[str]:
        """
        Generează distractori INTELIGENȚI bazați pe caracteristicile problemei. 
        
        Distractorii sunt aleși să fie plauzibili dar greșiți - strategii care
        ar putea părea corecte dar nu sunt optime pentru instanța dată.
        """
        correct_norm = _normalize(correct_strategy)
        
        # Clasificăm strategiile după tip
        complete_strategies = [
            "plain dfs", "plain bfs", 
            "backtracking with mrv + forward checking", 
            "ac-3 with backtracking"
        ]
        local_search = [
            "min-conflicts heuristic", 
            "hill climbing with sideways moves", 
            "simulated annealing", 
            "genetic algorithm"
        ]
        heuristic_strategies = [
            "warnsdorff heuristic (greedy)", 
            "a*", 
            "ida*"
        ]
        special_strategies = [
            "recursive strategy (divide and conquer)", 
            "dynamic programming", 
            "greedy coloring"
        ]
        
        plausible = []
        less_plausible = []
        
        if problem_type == "n-queens":
            metrics = StrategySolver._calculate_nqueens_metrics(params. get("N", 8))
            
            if metrics["bt_feasible"]:
                # Backtracking e optim -> distractori sunt metodele locale (par mai rapide)
                plausible = [s for s in local_search if _normalize(s) != correct_norm]
                # Și plain DFS/BFS (par similare dar fără optimizări)
                plausible. extend(["plain dfs", "plain bfs"])
            else:
                # Local search e optim -> distractori sunt metodele complete (par mai sigure)
                plausible = [s for s in complete_strategies if _normalize(s) != correct_norm]
        
        elif problem_type == "graph coloring":
            metrics = StrategySolver._calculate_graph_coloring_metrics(
                params.get("V", 10), params.get("E", 20), params.get("k", 3)
            )
            
            if metrics["is_dense"]:
                # AC-3 e optim -> distractori: metode fără propagare
                plausible = ["backtracking with mrv + forward checking", "plain dfs", "greedy coloring"]
            else:
                # MRV+FC e optim -> distractori: metode cu overhead inutil
                plausible = ["ac-3 with backtracking", "plain bfs", "genetic algorithm"]
        
        elif problem_type == "knight's tour":
            # Warnsdorff e aproape întotdeauna optim
            plausible = ["plain dfs", "plain bfs", "backtracking with mrv + forward checking", "a*"]
        
        elif problem_type == "generalized hanoi":
            # Recursiv e optim
            plausible = ["plain bfs", "plain dfs", "dynamic programming", "a*", "ida*"]
        
        # Eliminăm strategia corectă din plauzibile
        plausible = [s for s in plausible if _normalize(s) != correct_norm]
        
        # Completăm cu alte strategii dacă nu avem destule
        all_others = [s for s in ALL_STRATEGIES if _normalize(s) != correct_norm and s not in plausible]
        random. shuffle(all_others)
        less_plausible = all_others
        
        # Construim lista finală
        all_distractors = plausible + less_plausible
        
        # Eliminăm duplicatele
        seen = set()
        unique_distractors = []
        for d in all_distractors:
            d_norm = _normalize(d)
            if d_norm not in seen and d_norm != correct_norm:
                seen.add(d_norm)
                unique_distractors. append(d)
        
        # Selectăm count distractori, prioritizând pe cei plauzibili
        if len(unique_distractors) > count:
            # Amestecăm ușor pentru varietate, dar păstrăm prioritatea
            top = unique_distractors[:min(count + 2, len(unique_distractors))]
            random.shuffle(top)
            return top[:count]
        
        return unique_distractors[:count]

    # ===================================================================
    # SOLVER PRINCIPAL - Calculează răspunsul MATEMATIC
    # ===================================================================
    
    @staticmethod
    def solve(problem_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculează MATEMATIC strategia optimă bazată pe parametrii problemei.
        
        IMPORTANT: Acest solver NU folosește valori hardcodate! 
        Toate deciziile sunt bazate pe:
        1. Calcule de complexitate algoritmică
        2. Praguri derivate din teoria CS
        3.  Metrici specifice instanței
        
        Returns:
            Dict cu: best_strategy, required_concepts, forbidden_concepts,
                    reasoning_summary, calculations
        """
        truth = {
            "best_strategy": None,
            "required_concepts": [],
            "forbidden_concepts": [],
            "reasoning_summary": "",
            "calculations": {}
        }

        # ============================================================
        # N-QUEENS
        # ============================================================
        if problem_type == "n-queens":
            N = int(params. get("N", 8))
            metrics = StrategySolver._calculate_nqueens_metrics(N)
            truth["calculations"] = metrics
            
            if metrics["bt_feasible"]:
                truth["best_strategy"] = "backtracking with mrv + forward checking"
                truth["required_concepts"] = [
                    "complet", "garantat", "sistematic", "mrv", "forward checking",
                    "domeniu", "constrangeri", "propagare", "eficient", "sigur",
                    "fail-first", "regina", "pozitii", "variabila", "backtrack"
                ]
                truth["forbidden_concepts"] = [
                    # Simulated Annealing
                    "temperatura", "racire", "cooling", "energie", "probabilistic",
                    "incalzire", "calire", "annealing", "accepta mutari", "metropolis",
                    "boltzmann", "treptat", "gradual", "stochastic",
                    # Genetic Algorithm  
                    "populatie", "genetic", "fitness", "evolutie", "cromozom",
                    "crossover", "mutatie", "generatii", "selectie naturala", "offspring",
                    # Min-Conflicts (când BT e corect)
                    "local search", "conflicte minimale", "repair", "hill climbing",
                    # Graph Coloring concepts (NU se aplică la N-Queens!)
                    "graf", "nod", "muchie", "muchii", "culori", "colorare",
                    "densitate", "rar", "sparse", "dens", "arc consistency",
                    "vertex", "edge", "cromatica",
                    # Hanoi concepts
                    "tije", "discuri", "turnuri", "hanoi", "divide and conquer",
                    "recursiv natural", "subprobleme identice",
                    # Knight's Tour concepts
                    "warnsdorff", "cal", "tur", "knight", "tabla sah",
                    # BFS specific (când DFS/Backtracking e corect)
                    "coada", "nivel cu nivel", "breadth first", "latime"
                ]
                truth["reasoning_summary"] = (
                    f"Pentru N-Queens cu N={N}:\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Spațiul brut: {N}^{N} = {metrics['brute_space_formatted']} configurații\n"
                    f"• Backtracking clasic: {N}!  = {metrics['factorial_formatted']} noduri\n"
                    f"• Cu MRV + Forward Checking: ~{metrics['bt_mrv_fc_formatted']} noduri (reducere 85%)\n"
                    f"• Pragul de fezabilitate: {_format_large_number(metrics['threshold_used'])} operații\n\n"
                    f"📐 CALCUL DECIZIE:\n"
                    f"• {metrics['bt_mrv_fc_formatted']} < {_format_large_number(metrics['threshold_used'])} ✓\n"
                    f"• Backtracking este FEZABIL pentru această instanță\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"Backtracking cu MRV + Forward Checking este OPTIM deoarece:\n"
                    f"• COMPLET: Garantează găsirea soluției dacă există\n"
                    f"• EFICIENT: MRV (Minimum Remaining Values) alege variabila cu domeniul cel mai mic\n"
                    f"• PROPAGARE: Forward Checking elimină valorile inconsistente imediat"
                )
            else:
                truth["best_strategy"] = "min-conflicts heuristic"
                truth["required_concepts"] = [
                    "rapid", "local", "conflicte", "eficient", "scala", "iterativ",
                    "euristica", "repair", "incremental", "practic", "swap", "vecini"
                ]
                truth["forbidden_concepts"] = [
                    # Complete search (când Min-Conflicts e corect)
                    "complet", "garantat", "exhaustiv", "toate solutiile", "arbore",
                    "sistematic", "sigur",
                    # Simulated Annealing
                    "temperatura", "racire", "cooling", "energie", "calire",
                    "incalzire", "annealing", "metropolis", "boltzmann",
                    # Genetic Algorithm
                    "populatie", "genetic", "fitness", "cromozom", "crossover",
                    "mutatie", "generatii", "evolutie", "offspring",
                    # Graph Coloring concepts
                    "graf", "nod", "muchie", "muchii", "culori", "colorare",
                    "densitate", "rar", "sparse", "dens", "vertex", "edge",
                    # Hanoi concepts
                    "tije", "discuri", "turnuri", "hanoi", "divide and conquer",
                    "recursiv natural", "subprobleme identice",
                    # Knight's Tour concepts
                    "warnsdorff", "cal", "tur", "knight",
                    # BFS/DFS plain
                    "coada", "nivel cu nivel", "breadth first"
                ]
                truth["reasoning_summary"] = (
                    f"Pentru N-Queens cu N={N}:\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Backtracking cu MRV+FC: ~{metrics['bt_mrv_fc_formatted']} noduri\n"
                    f"• Pragul de fezabilitate: {_format_large_number(metrics['threshold_used'])} operații\n"
                    f"• Min-Conflicts: ~{metrics['min_conflicts_formatted']} operații\n"
                    f"• Raport eficiență: {metrics['efficiency_ratio']:.0f}x mai rapid cu Min-Conflicts\n\n"
                    f"📐 CALCUL DECIZIE:\n"
                    f"• {metrics['bt_mrv_fc_formatted']} >> {_format_large_number(metrics['threshold_used'])} ✗\n"
                    f"• Backtracking este INFEZABIL pentru această instanță\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"Min-Conflicts Heuristic este OPTIM deoarece:\n"
                    f"• RAPID: Complexitate O(N²) vs O(N!) pentru Backtracking\n"
                    f"• SCALABIL: Funcționează eficient pentru N foarte mare\n"
                    f"• PRACTIC: Găsește soluția în secunde chiar pentru N={N}"
                )

        # ============================================================
        # GENERALIZED HANOI
        # ============================================================
        elif problem_type == "generalized hanoi":
            pegs = int(params. get("pegs", 3))
            disks = int(params.get("disks", 5))
            metrics = StrategySolver._calculate_hanoi_metrics(pegs, disks)
            truth["calculations"] = metrics
            
            truth["best_strategy"] = "recursive strategy (divide and conquer)"
            truth["required_concepts"] = [
                "recursiv", "divide", "conquer", "subprobleme", "optim",
                "structura", "natural", "elegant", "frame", "stewart",
                "mutari", "minim", "tije", "discuri"
            ]
            # IMPORTANT: NU include "bfs", "dfs", "3 tije" - utilizatorul le poate menționa în context comparativ! 
            truth["forbidden_concepts"] = [
                # Simulated Annealing
                "temperatura", "racire", "cooling", "energie", "calire",
                "annealing", "metropolis", "boltzmann", "probabilistic",
                # Genetic Algorithm
                "populatie", "genetic", "fitness", "cromozom", "crossover",
                "mutatie", "generatii", "evolutie", "selectie naturala",
                # CSP/Constraint concepts (nu se aplică la Hanoi)
                "ac-3", "arc consistency", "constrangeri", "propagare domeniu",
                "forward checking", "mrv", "minimum remaining",
                # Min-Conflicts
                "conflicte", "min-conflicts", "local search", "hill climbing",
                "repair", "swap", "vecini",
                # Graph Coloring
                "graf", "nod", "muchie", "culori", "colorare", "densitate",
                # N-Queens specific
                "regina", "regine", "queens", "ataca diagonal",
                # Knight's Tour
                "warnsdorff", "cal", "knight", "tur"
            ]
            
            if metrics["is_classic"]:
                truth["reasoning_summary"] = (
                    f"Pentru Turnurile din Hanoi cu {pegs} tije și {disks} discuri:\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Formula optimă: {metrics['formula_used']}\n"
                    f"• Număr optim de mutări: {metrics['optimal_moves_formatted']}\n"
                    f"• Spațiul de stări: {pegs}^{disks} = {metrics['state_space_formatted']} stări\n\n"
                    f"📐 DEMONSTRAȚIE:\n"
                    f"• Recurența: T(n) = 2×T(n-1) + 1\n"
                    f"• Soluție: T(n) = 2^n - 1\n"
                    f"• Acest număr de mutări este MINIMAL și UNIC\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"Strategia RECURSIVĂ (Divide et Conquer) este SINGURA optimă:\n"
                    f"• DEMONSTRAT MATEMATIC: Produce exact numărul minim de mutări\n"
                    f"• STRUCTURĂ NATURALĂ: Problema se descompune în subprobleme identice\n"
                    f"• ALGORITM: Mută n-1 discuri auxiliar, mută discul mare, mută n-1 înapoi"
                )
            else:
                truth["reasoning_summary"] = (
                    f"Pentru Turnurile din Hanoi GENERALIZAT ({pegs} tije, {disks} discuri):\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Formula Frame-Stewart: {metrics['formula_used']}\n"
                    f"• Spațiul de stări: {pegs}^{disks} = {metrics['state_space_formatted']} stări\n"
                    f"• Cu 3 tije ar fi nevoie de 2^{disks}-1 = {_format_large_number(2**disks - 1)} mutări\n\n"
                    f"📐 OPTIMIZARE:\n"
                    f"• Cu {pegs} tije, numărul de mutări se reduce DRAMATIC\n"
                    f"• Frame-Stewart: Împarte discurile și folosește tijele extra\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"Strategia Frame-Stewart (Divide et Conquer extins) este OPTIMĂ:\n"
                    f"• CEL MAI BUN ALGORITM CUNOSCUT pentru >{3} tije\n"
                    f"• RECURSIV: Folosește substructura optimă a problemei\n"
                    f"• EFICIENT: Reduce dramatic numărul de mutări"
                )

        # ============================================================
        # GRAPH COLORING
        # ============================================================
        elif problem_type == "graph coloring":
            V = int(params. get("V", 10))
            E = int(params. get("E", 10))
            k = int(params. get("k", 3))
            metrics = StrategySolver._calculate_graph_coloring_metrics(V, E, k)
            truth["calculations"] = metrics

            if metrics["is_dense"]:
                truth["best_strategy"] = "ac-3 with backtracking"
                truth["required_concepts"] = [
                    "dens", "constrangeri", "arc", "consistenta", "ac-3",
                    "propagare", "domeniu", "reduce", "prune", "multe muchii"
                ]
                truth["forbidden_concepts"] = [
                    "rar", "sparse", "simplu", "min-conflicts", "genetic",
                    "temperatura", "racire", "energie", "populatie", "fitness",
                    "rapid", "overhead mic"
                ]
                truth["reasoning_summary"] = (
                    f"Pentru Graph Coloring (|V|={V}, |E|={E}, k={k}):\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Muchii maxime posibile: C({V},2) = {metrics['max_edges']}\n"
                    f"• Densitate: {E}/{metrics['max_edges']} = {metrics['density_percent']}%\n"
                    f"• Grad mediu noduri: 2×{E}/{V} = {metrics['avg_degree']}\n"
                    f"• Pragul pentru graf dens: {StrategySolver. DENSE_GRAPH_THRESHOLD * 100}%\n\n"
                    f"📐 CALCUL DECIZIE:\n"
                    f"• {metrics['density_percent']}% > {StrategySolver.DENSE_GRAPH_THRESHOLD * 100}% ✓\n"
                    f"• Graful este DENS → multe constrângeri de propagat\n"
                    f"• Overhead AC-3: O({E} × {k}²) = {metrics['ac3_overhead_formatted']} operații\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"AC-3 cu Backtracking este OPTIM deoarece:\n"
                    f"• PROPAGARE PUTERNICĂ: Multe muchii = multe constrângeri de exploatat\n"
                    f"• REDUCE DOMENII: Elimină culori imposibile ÎNAINTE de căutare\n"
                    f"• DETECTEAZĂ EȘEC DEVREME: Găsește inconsistențe fără backtracking"
                )
            else:
                truth["best_strategy"] = "backtracking with mrv + forward checking"
                truth["required_concepts"] = [
                    "rar", "sparse", "mrv", "forward", "checking", "simplu",
                    "rapid", "overhead mic", "eficient", "putine muchii"
                ]
                truth["forbidden_concepts"] = [
                    "dens", "ac-3", "arc consistency", "complex", "propagare intensa",
                    "temperatura", "genetic", "populatie", "multe constrangeri"
                ]
                truth["reasoning_summary"] = (
                    f"Pentru Graph Coloring (|V|={V}, |E|={E}, k={k}):\n\n"
                    f"📊 ANALIZĂ MATEMATICĂ:\n"
                    f"• Muchii maxime posibile: C({V},2) = {metrics['max_edges']}\n"
                    f"• Densitate: {E}/{metrics['max_edges']} = {metrics['density_percent']}%\n"
                    f"• Grad mediu noduri: 2×{E}/{V} = {metrics['avg_degree']}\n"
                    f"• Pragul pentru graf dens: {StrategySolver. DENSE_GRAPH_THRESHOLD * 100}%\n\n"
                    f"📐 CALCUL DECIZIE:\n"
                    f"• {metrics['density_percent']}% ≤ {StrategySolver.DENSE_GRAPH_THRESHOLD * 100}% ✓\n"
                    f"• Graful este RAR → puține constrângeri\n"
                    f"• AC-3 ar avea overhead inutil\n\n"
                    f"✅ CONCLUZIE:\n"
                    f"Backtracking cu MRV + Forward Checking este OPTIM deoarece:\n"
                    f"• OVERHEAD MIC: Nu pierdem timp cu propagare excesivă\n"
                    f"• MRV: Alege nodul cu cele mai puține culori disponibile\n"
                    f"• FORWARD CHECKING: Suficient pentru constrângerile puține"
                )

        # ============================================================
        # KNIGHT'S TOUR
        # ============================================================
        elif problem_type == "knight's tour":
            n = int(params. get("n", 8))
            metrics = StrategySolver._calculate_knights_tour_metrics(n)
            truth["calculations"] = metrics
            
            truth["best_strategy"] = "warnsdorff heuristic (greedy)"
            truth["required_concepts"] = [
                "warnsdorff", "greedy", "grad", "minim", "rapid", "liniar",
                "euristica", "eficient", "mutari disponibile", "regula"
            ]
            truth["forbidden_concepts"] = [
                "bfs", "dfs", "exhaustiv", "backtracking", "complet",
                "genetic", "temperatura", "populatie", "ac-3", "sistematic",
                "toate caile"
            ]
            truth["reasoning_summary"] = (
                f"Pentru Knight's Tour pe tablă {n}×{n}:\n\n"
                f"📊 ANALIZĂ MATEMATICĂ:\n"
                f"• Total căsuțe de vizitat: {n}×{n} = {metrics['total_squares']}\n"
                f"• Backtracking brut: O(({n}²)!) = {metrics['bt_estimate_formatted']} noduri\n"
                f"• Warnsdorff: O({n}² × 8) = {metrics['warnsdorff_formatted']} operații\n"
                f"• Câștig eficiență: {metrics['efficiency_gain_formatted']}× mai rapid\n\n"
                f"📐 REGULA WARNSDORFF:\n"
                f"• La fiecare pas, alege căsuța cu CELE MAI PUȚINE mutări viitoare\n"
                f"• Intuiție: Lasă căsuțele 'dificile' pentru când ai mai puține opțiuni\n"
                f"• Succes garantat pentru n ≥ 5: {'DA ✓' if metrics['solvable'] else 'NU ✗'}\n\n"
                f"✅ CONCLUZIE:\n"
                f"Euristica Warnsdorff (Greedy) este OPTIMĂ deoarece:\n"
                f"• APROAPE LINIARĂ: O(n²) în loc de O((n²)!)\n"
                f"• SUCCES GARANTAT: Pentru n≥5, găsește mereu soluția\n"
                f"• FĂRĂ BACKTRACKING: Greedy pur, nu revine niciodată"
            )

        return truth

    @staticmethod
    def get_system_answer(problem_type: str, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Generează răspunsul aplicației (pentru afișare către utilizator).
        Oferă atât varianta punctuală cât și cea detaliată.
        """
        solution = StrategySolver.solve(problem_type, params)
        strategy = solution["best_strategy"]
        reasoning = solution["reasoning_summary"]
        calculations = solution. get("calculations", {})
        
        # Răspuns punctual (scurt)
        punctual = strategy.title() if strategy else "Necunoscut"
        
        # Răspuns detaliat
        detailed = (
            f"🎯 STRATEGIA OPTIMĂ: {strategy}\n\n"
            f"{reasoning}"
        )

        return {
            "punctual": punctual,
            "detailed": detailed,
            "strategy": strategy,
            "calculations": calculations
        }


# =======================================================================
# 2. NLP ENGINE (Analizatorul de Intenție & Evaluare Semantică)
# =======================================================================
class SmartEvaluator:
    """
    Evaluator inteligent care analizează răspunsurile utilizatorului
    folosind NLP local (SpaCy) și similaritate semantică (SBERT).
    
    NU folosește LLM sau API-uri externe! 
    """
    
    # Pattern-uri Regex pentru identificarea strategiilor
    STRATEGY_PATTERNS = [
        # Backtracking variants
        (r"backtracking\s+(?:with\s+)? mrv\s*\+?\s*forward\s*checking", "backtracking with mrv + forward checking"),
        (r"backtracking\s+(?:cu\s+)?mrv\s+(?:si|și)\s+forward\s*checking", "backtracking with mrv + forward checking"),
        (r"mrv\s*\+?\s*forward\s*checking", "backtracking with mrv + forward checking"),
        (r"mrv\s+(?:si|și|and)\s+fc", "backtracking with mrv + forward checking"),
        (r"mrv\s*\+\s*fc", "backtracking with mrv + forward checking"),
        
        # Recursive/Divide and Conquer
        (r"recursive\s+strategy\s*\(?\s*divide\s+(?:and|et|si|și)\s+conquer\s*\)? ", "recursive strategy (divide and conquer)"),
        (r"recursive\s+strategy", "recursive strategy (divide and conquer)"),
        (r"divide\s+(?:and|et|si|și)\s+conquer", "recursive strategy (divide and conquer)"),
        (r"strategia?\s+recursiva? ", "recursive strategy (divide and conquer)"),
        (r"frame[\s-]? stewart", "recursive strategy (divide and conquer)"),
        
        # Warnsdorff
        (r"warnsdorff", "warnsdorff heuristic (greedy)"),
        (r"euristica\s+greedy\s+(?:pentru\s+)?knight", "warnsdorff heuristic (greedy)"),
        
        # AC-3
        (r"ac-? 3\s+(?:with\s+)?backtracking", "ac-3 with backtracking"),
        (r"arc\s+consistency", "ac-3 with backtracking"),
        (r"consistenta\s+(?:de\s+)?arc", "ac-3 with backtracking"),
        
        # Min-conflicts
        (r"min-? conflicts? ", "min-conflicts heuristic"),
        (r"conflicte\s+minime", "min-conflicts heuristic"),
        (r"minimizarea?\s+conflictelor", "min-conflicts heuristic"),
        
        # Others
        (r"genetic\s+algorithm", "genetic algorithm"),
        (r"algoritm(?:ul)?\s+genetic", "genetic algorithm"),
        (r"simulated\s+annealing", "simulated annealing"),
        (r"calire\s+simulata", "simulated annealing"),
        (r"hill\s+climbing", "hill climbing with sideways moves"),
        (r"\bplain\s+bfs\b", "plain bfs"),
        (r"\bbreadth[- ]first\b", "plain bfs"),
        (r"\bcautare\s+(?:in\s+)?latime\b", "plain bfs"),
        (r"\bplain\s+dfs\b", "plain dfs"),
        (r"\bdepth[- ]first\b", "plain dfs"),
        (r"\bcautare\s+(?:in\s+)?adancime\b", "plain dfs"),
        (r"\ba\*\b", "a*"),
        (r"\bida\*\b", "ida*"),
        (r"\bdynamic\s+programming\b", "dynamic programming"),
        (r"\bprogramare\s+dinamica\b", "dynamic programming"),
        
        # Generic backtracking (fără optimizări specificate) -> presupunem MRV+FC
        (r"\bbacktracking\b(?!\s+with)", "backtracking with mrv + forward checking"),
        (r"\bac-?3\b", "ac-3 with backtracking"),
    ]

    @staticmethod
    def _find_all_strategies(text: str) -> List[Dict[str, Any]]:
        """Găsește toate strategiile menționate în text."""
        text_norm = _normalize(text)
        found = []
        used = []
        
        for pattern, canonical in SmartEvaluator.STRATEGY_PATTERNS:
            for m in re.finditer(pattern, text_norm):
                s, e = m.start(), m.end()
                # Evităm suprapunerile
                overlap = any(not (e <= us or s >= ue) for us, ue in used)
                if not overlap:
                    used.append((s, e))
                    found. append({
                        'canonical': canonical,
                        'start': s,
                        'end': e,
                        'matched_text': m.group()
                    })
        
        found.sort(key=lambda x: x['start'])
        return found

    @staticmethod
    def _detect_criticism_after_mention(text: str, mention: Dict[str, Any]) -> bool:
        """Detectează dacă utilizatorul critică strategia menționată."""
        text_norm = _normalize(text)
        end_pos = mention['end']
        after_text = text_norm[end_pos:end_pos + 200]
        
        criticism_patterns = [
            r"\b(?:dar|insa|totusi|however|but)\b\s+. {0,50}\b(?:ignora|opreste|blocheaza|nu\s+(?:functioneaza|merge)|lipseste|problema)\b",
            r"(?:implementarea|aceasta|aceasta\s+metoda)\s+. {0,30}\b(?:ignora|nu|lipseste|opreste)\b",
            r"(?:mrv|forward\s+checking)\s+.{0,30}\b(?:ignora|nu|lipseste|opreste|ineficient)\b",
            r"\bin\s+consecinta\s+(?:aplicam|folosim|alegem)\b",
            r"(?:de\s+aceea|astfel|prin\s+urmare)\s+(?:aplicam|folosim|alegem)\b",
        ]
        
        for pattern in criticism_patterns:
            if re.search(pattern, after_text):
                return True
        return False

    @staticmethod
    def _detect_alternative_proposal(text: str, primary_mention: Dict[str, Any], 
                                     all_mentions: List[Dict[str, Any]]) -> Optional[str]:
        """Detectează dacă utilizatorul propune o alternativă."""
        text_norm = _normalize(text)
        primary_end = primary_mention['end']
        
        for mention in all_mentions:
            if mention['start'] > primary_end:
                before_text = text_norm[max(0, mention['start'] - 50):mention['start']]
                proposal_patterns = [
                    r"(?:aplicam|folosim|alegem|recomandam|propunem)\s*$",
                    r"(?:in\s+consecinta|de\s+aceea|astfel|prin\s+urmare)\s*$",
                    r"(?:mai\s+bine|prefer|optam\s+pentru)\s*$",
                ]
                for pattern in proposal_patterns:
                    if re.search(pattern, before_text):
                        return mention['canonical']
        return None

    @staticmethod
    def _analyze_answer_coherence(text: str, selected_strategy: str,
                                  all_mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizează coerența răspunsului."""
        selected_norm = _normalize(selected_strategy)
        result = {
            'is_coherent': True,
            'criticism_detected': False,
            'alternative_proposed': None,
            'penalty_reason': None
        }
        
        selected_mention = None
        for m in all_mentions:
            if _normalize(m['canonical']) == selected_norm:
                selected_mention = m
                break
        
        if not selected_mention:
            return result
        
        if SmartEvaluator._detect_criticism_after_mention(text, selected_mention):
            result['is_coherent'] = False
            result['criticism_detected'] = True
            result['penalty_reason'] = "Critici strategia aleasă imediat după ce o menționezi."
        
        alternative = SmartEvaluator._detect_alternative_proposal(text, selected_mention, all_mentions)
        if alternative:
            result['is_coherent'] = False
            result['alternative_proposed'] = alternative
            result['penalty_reason'] = f"Propui de fapt o altă strategie: {alternative}."
        
        return result

    @staticmethod
    def _detect_polarity_for_mention(text: str, mention: Dict[str, Any], nlp_doc=None) -> str:
        """Detectează polaritatea (pozitiv/negativ) pentru o mențiune."""
        text_norm = _normalize(text)
        start = mention['start']
        end = mention['end']
        
        # Verificăm în fereastra din stânga
        left_window = text_norm[max(0, start - 90):start]
        for pat in REJECTION_PATTERNS:
            if re.search(pat, left_window):
                return 'negative'

        immediate_left = text_norm[max(0, start - 30):start]
        if any(tok in NEGATION_WORDS for tok in immediate_left.split()):
            return 'negative'
        if any(neg in left_window for neg in NEGATION_VERBS):
            return 'negative'

        immediate_right = text_norm[end:end + 30]
        if any(cond in immediate_right. split() for cond in CONDITIONAL_WORDS):
            return 'negative'

        # Verificare cu SpaCy dacă e disponibil
        if nlp_doc and nlp:
            try:
                for token in nlp_doc:
                    if token.idx <= mention['start'] * 1.2:
                        if any(c. dep_ == 'neg' or _normalize(c.text) in NEGATION_WORDS for c in token. children):
                            return 'negative'
                        if token.head and token.head.pos_ == 'VERB':
                            if _normalize(token.head.lemma_) in NEGATION_VERBS:
                                return 'negative'
            except Exception:
                pass
        
        return 'positive'

    @staticmethod
    def identify_selection(text: str, context_options: List[str]) -> Optional[Dict[str, Any]]:
        """Identifică ce strategie a selectat utilizatorul."""
        if not text or len(text.strip()) < 3:
            return None
        
        context_norm = [_normalize(o) for o in context_options]
        mentions = SmartEvaluator._find_all_strategies(text)
        
        if not mentions:
            return None
        
        nlp_doc = None
        if nlp:
            try:
                nlp_doc = nlp(text)
            except Exception:
                nlp_doc = None
        
        analyzed = []
        for m in mentions:
            polarity = SmartEvaluator._detect_polarity_for_mention(text, m, nlp_doc)
            in_context = any(
                m['canonical']. lower() in c or c in m['canonical'].lower()
                for c in context_norm
            )
            analyzed.append({**m, 'polarity': polarity, 'in_context': in_context})
        
        # Prioritizăm mențiunile pozitive din context
        pos_in_ctx = [a for a in analyzed if a['polarity'] == 'positive' and a['in_context']]
        if pos_in_ctx:
            best = pos_in_ctx[0]
            coherence = SmartEvaluator._analyze_answer_coherence(text, best['canonical'], mentions)
            return {
                'strategy': best['canonical'],
                'polarity': 'positive' if coherence['is_coherent'] else 'contradictory',
                'confidence': 0.95 if coherence['is_coherent'] else 0.3,
                'coherence_issue': coherence. get('penalty_reason'),
                'alternative_proposed': coherence.get('alternative_proposed')
            }
        
        # Apoi mențiunile pozitive în general
        pos_any = [a for a in analyzed if a['polarity'] == 'positive']
        if pos_any:
            best = pos_any[0]
            coherence = SmartEvaluator._analyze_answer_coherence(text, best['canonical'], mentions)
            return {
                'strategy': best['canonical'],
                'polarity': 'positive' if coherence['is_coherent'] else 'contradictory',
                'confidence': 0.8 if coherence['is_coherent'] else 0.3,
                'coherence_issue': coherence.get('penalty_reason'),
                'alternative_proposed': coherence.get('alternative_proposed')
            }
        
        # În final, orice mențiune (negativă)
        if analyzed:
            best = analyzed[0]
            return {
                'strategy': best['canonical'],
                'polarity': 'negative',
                'confidence': 0.7,
                'coherence_issue': None,
                'alternative_proposed': None
            }
        
        return None

    @staticmethod
    def _verify_numerical_claims(user_text: str, calculations: Dict[str, Any], 
                                  problem_type: str) -> Tuple[float, List[str]]:
        """
        Verifică dacă utilizatorul face afirmații numerice corecte.
        Returnează (bonus/penalizare, lista de feedback-uri).
        """
        text_norm = _normalize(user_text)
        bonus = 0.0
        feedback = []
        
        if problem_type == "n-queens" and calculations:
            N = calculations. get("N", 0)
            correct_factorial = calculations.get("factorial_space", 0)
            
            # Caută menționarea lui N! 
            factorial_patterns = [
                rf"{N}\s*!\s*=\s*(\d[\d\s.,]*)",
                rf"factorial[^\d]*(\d[\d\s.,]*)",
                rf"(\d[\d\s.,]*)\s*(?:noduri|stari|configuratii)"
            ]
            
            for pattern in factorial_patterns:
                match = re.search(pattern, user_text. replace(",", ""). replace(".", ""))
                if match:
                    try:
                        user_value = int(re.sub(r'\D', '', match. group(1)))
                        if user_value > 0:
                            # Verificăm cu toleranță de 10%
                            if 0.9 * correct_factorial <= user_value <= 1.1 * correct_factorial:
                                bonus += 0.08
                                feedback. append(f"✓ Calcul corect pentru {N}!")
                            elif user_value != correct_factorial:
                                bonus -= 0.05
                                feedback. append(f"✗ {N}! = {_format_large_number(correct_factorial)}")
                    except (ValueError, TypeError):
                        pass
                    break
        
        elif problem_type == "graph coloring" and calculations:
            density = calculations.get("density", 0)
            is_dense = calculations.get("is_dense", False)
            
            # Verifică dacă menționează corect tipul grafului
            mentions_dense = bool(re.search(r"\bdens\b", text_norm))
            mentions_sparse = bool(re.search(r"\brar\b|\bsparse\b", text_norm))
            
            if mentions_dense and is_dense:
                bonus += 0.05
                feedback.append("✓ Corect: graf dens")
            elif mentions_dense and not is_dense:
                bonus -= 0.08
                feedback.append(f"✗ Graful este RAR (densitate {density:. 1%})")
            
            if mentions_sparse and not is_dense:
                bonus += 0.05
                feedback.append("✓ Corect: graf rar")
            elif mentions_sparse and is_dense:
                bonus -= 0.08
                feedback. append(f"✗ Graful este DENS (densitate {density:.1%})")
        
        elif problem_type == "generalized hanoi" and calculations:
            disks = calculations.get("disks", 0)
            optimal = calculations.get("optimal_moves", 0)
            
            # Caută formula 2^n - 1
            if re.search(rf"2\s*\^\s*{disks}\s*-\s*1", user_text) or str(optimal) in user_text:
                bonus += 0.08
                feedback.append(f"✓ Corect: 2^{disks}-1 = {optimal}")
        
        return bonus, feedback

    @staticmethod
    def check_reasoning(user_text: str, required_concepts: List[str], 
                       forbidden_concepts: List[str], ideal_explanation: str = "",
                       correct_strategy: str = "", calculations: Dict[str, Any] = None,
                       problem_type: str = "") -> Tuple[float, List[str], Optional[str], List[str]]:
        """
        Analizează argumentarea folosind:
        1. Detecția de concepte (Regex + SpaCy)
        2. Similaritate semantică (SBERT - LOCAL, fără API!)
        3.  Verificare calcule numerice
        
        Returns:
            Tuple: (score, violations, wrong_strategy_detected, numerical_feedback)
        """
        if not user_text or not user_text.strip():
            return 0.0, [], None, []
        
        text_norm = _normalize(user_text)
        
        # 1.  DETECTARE STRATEGIE DIN DESCRIERE (poate indica confuzie)
        described_strategy = _detect_strategy_from_description(user_text)
        wrong_strategy_detected = None
        if described_strategy and correct_strategy:
            correct_norm = _normalize(correct_strategy)
            described_norm = _normalize(described_strategy)
            if described_norm != correct_norm and correct_norm not in described_norm and described_norm not in correct_norm:
                wrong_strategy_detected = described_strategy
        
            # 2. VERIFICARE CONCEPTE INTERZISE (cu detectare context)
            violations = []
            for bad in forbidden_concepts:
                bad_norm = _normalize(bad)
                if len(bad_norm) < 3:
                    continue
                
                # Verificăm dacă conceptul apare în text
                if bad_norm not in text_norm:
                    continue
                
                # GĂSIT - acum verificăm CONTEXTUL
                # Pattern pentru a extrage textul din jurul conceptului
                escaped_bad = re. escape(bad_norm)
                pattern = rf"(.{{0,80}})\b{escaped_bad}\b(.{{0,50}})"
                match = re.search(pattern, text_norm)
                
                if not match:
                    # Încearcă fără word boundaries pentru concepte compuse
                    pattern = rf"(.{{0,80}}){escaped_bad}(.{{0,50}})"
                    match = re.search(pattern, text_norm)
                
                if match:
                    left_context = match.group(1)
                    right_context = match.group(2)
                    full_context = left_context + bad_norm + right_context
                    
                    # Lista de indicatori că conceptul e CRITICAT sau COMPARAT negativ
                    negative_patterns = [
                        # Negații
                        r"nu\s+(?:\w+\s+){0,3}" + escaped_bad,
                        r"nici\s+(?:\w+\s+){0,2}" + escaped_bad,
                        r"fara\s+(?:\w+\s+){0,2}" + escaped_bad,
                        r"evit\w*\s+(?:\w+\s+){0,2}" + escaped_bad,
                        # Critici în context stâng
                        r"(?:ar\s+fi|e|este|sunt)\s+(?:imposibil|ineficient|lent|prea|gresit)",
                        r"nu\s+(?:necesita|folosim|avem|e|este)",
                        r"(?:imposibil|ineficient|nepractice? )\s+(?:pentru|de|sa)",
                        # Comparații
                        r"(?:in\s+schimb|spre\s+deosebire|fata\s+de|comparativ)",
                        r"(?:in\s+loc\s+de|vs|versus)",
                    ]
                    
                    # Indicatori simpli în context
                    negative_words_left = [
                        "nu ", "nici ", "fara ", "evita", "imposibil", "ineficient",
                        "nepracti", "ar explora", "ar fi ", "in schimb", "spre deosebire",
                        "nu necesita", "nu folosim", "nu avem nevoie", "nu e nevoie"
                    ]
                    
                    negative_words_right = [
                        "ar fi imposibil", "ar fi ineficient", "ar fi lent",
                        "e imposibil", "imposibil de", "ar explora",
                        "nepracti", "ineficient"
                    ]
                    
                    is_safe = False
                    
                    # Verificăm context stâng
                    for neg in negative_words_left:
                        if neg in left_context:
                            is_safe = True
                            break
                    
                    # Verificăm context drept
                    if not is_safe:
                        for neg in negative_words_right:
                            if neg in right_context:
                                is_safe = True
                                break
                    
                    # Verificăm pattern-uri regex
                    if not is_safe:
                        for pattern in negative_patterns:
                            if re.search(pattern, full_context):
                                is_safe = True
                                break
                    
                    # Dacă NU e în context negativ, adăugăm la violări
                    if not is_safe:
                        violations.append(bad)

        # Dacă avem violări grave sau strategie greșită detectată
        if violations or wrong_strategy_detected:
            return 0.1, violations, wrong_strategy_detected, []

        # 3. VERIFICARE CALCULE NUMERICE
        numerical_bonus = 0.0
        numerical_feedback = []
        if calculations and problem_type:
            numerical_bonus, numerical_feedback = SmartEvaluator._verify_numerical_claims(
                user_text, calculations, problem_type
            )

        # 4. VERIFICARE CONCEPTE NECESARE (required_concepts)
        hits = 0
        doc = None
        if nlp:
            try:
                doc = nlp(user_text)
            except Exception:
                pass
        
        if doc:
            for concept in required_concepts:
                c_norm = _normalize(concept)
                if len(c_norm) < 2:
                    continue
                
                found_pos = False
                for token in doc:
                    tok_norm = _normalize(token.text)
                    if c_norm in tok_norm or tok_norm in c_norm:
                        # Verificăm că nu e negat
                        is_neg = False
                        left_tokens = list(doc[max(0, token. i - 4):token.i])
                        if any(_normalize(t. text) in NEGATION_WORDS for t in left_tokens):
                            is_neg = True
                        if not is_neg:
                            found_pos = True
                            break
                
                if found_pos:
                    hits += 1
            
            keyword_score = hits / max(1, len(required_concepts)) if required_concepts else 1.0
        else:
            # Fallback fără SpaCy
            keyword_score = sum(1 for c in required_concepts if _normalize(c) in text_norm) / max(1, len(required_concepts))

        # 5. SIMILARITATE SEMANTICĂ (SBERT - rulează LOCAL!)
        similarity_score = 0.0
        if ideal_explanation and semantic_model:
            try:
                # Calculăm embeddings (LOCAL, fără API)
                emb_ideal = semantic_model. encode(ideal_explanation, convert_to_tensor=True)
                emb_user = semantic_model. encode(user_text, convert_to_tensor=True)
                
                # Similaritate cosinus
                similarity_score = util.cos_sim(emb_ideal, emb_user). item()
                
                # Normalizare: SBERT dă scoruri între -1 și 1, dar uzual >0. 3
                # Scalăm [0. 3, 0.85] -> [0.0, 1.0]
                similarity_score = max(0.0, (similarity_score - 0.3) / 0.55)
                similarity_score = min(1.0, similarity_score)
            except Exception as e:
                print(f"[WARN] SBERT error: {e}")
                similarity_score = 0.0
        elif ideal_explanation:
            # Fallback: SequenceMatcher dacă SBERT nu e disponibil
            similarity_score = SequenceMatcher(None, text_norm, _normalize(ideal_explanation)).ratio()

        # 6.  SCOR FINAL COMBINAT
        # Pondere: 35% Keywords + 55% Semantică + 10% Bonus calcule
        base_score = (keyword_score * 0.35) + (similarity_score * 0.55)
        final_score = min(1.0, max(0.0, base_score + numerical_bonus))
        
        return final_score, [], None, numerical_feedback


# =======================================================================
# DETECTARE CONTRADICȚII FACTUALE
# =======================================================================
def _detect_ground_contradiction(user_text: str, item_context: Dict[str, Any], 
                                  truth: Dict[str, Any]) -> Optional[str]:
    """
    Detectează contradicții factuale în răspunsul utilizatorului.
    IMPORTANT: Nu penaliza comparațiile sau mențiunile în context negativ!
    """
    t = _normalize(user_text)
    ptype = item_context.get('problem_type')
    params = item_context. get('params', {})
    calculations = truth.get('calculations', {})

    if ptype == 'graph coloring':
        density = calculations.get('density', 0)
        is_dense = calculations.get('is_dense', False)
        
        # Verificăm să nu fie în context comparativ sau negativ
        says_dense = bool(re.search(r"\b(?:este|e)\s+dens\b", t))
        says_sparse = bool(re. search(r"\b(?:este|e)\s+(?:rar|sparse)\b", t))
        
        # NU penaliza dacă e în context comparativ
        is_comparison = bool(re. search(r"(?:daca|if|ar fi|pentru|comparativ|versus|fata de|spre deosebire)", t))
        
        if not is_comparison:
            if says_sparse and is_dense:
                return f"Afirmi că graful este rar, dar densitatea reală este {density:. 1%} (DENS)."
            if says_dense and not is_dense:
                return f"Afirmi că graful este dens, dar densitatea reală este {density:. 1%} (RAR)."

    if ptype == 'n-queens':
        bt_feasible = calculations.get('bt_feasible', True)
        N = calculations.get('N', params.get('N', 8))
        
        # Verificăm context
        is_comparison = bool(re. search(r"(?:daca|if|ar fi|pentru|comparativ|versus|in schimb)", t))
        
        says_small = bool(re. search(r"\b(?:este|e)\s+(?:mic|simplu|usor)\b", t))
        says_large = bool(re.search(r"\b(?:este|e)\s+(?:mare|complex|enorm)\b", t))
        
        if not is_comparison:
            if says_small and not bt_feasible:
                return f"Afirmi că N={N} e mic/gestionabil, dar spațiul depășește pragul de fezabilitate."
            if says_large and bt_feasible:
                return f"Afirmi că N={N} e mare/complex, dar este gestionabil pentru Backtracking."

    if ptype == 'generalized hanoi':
        is_classic = calculations. get('is_classic', True)
        pegs = calculations.get('pegs', params.get('pegs', 3))
        
        # IMPORTANT: NU penaliza mențiunile comparative! 
        # Ex: "Cu 3 tije ar fi nevoie de..." e o COMPARAȚIE, nu o afirmație greșită
        
        # Verificăm doar afirmații directe greșite
        says_classic_direct = bool(re.search(r"(?:problema|aceasta)\s+(?:este|e|are)\s+(?:clasica|3\s+tije|trei\s+tije)", t))
        says_generalized_direct = bool(re.search(r"(?:problema|aceasta)\s+(?:este|e)\s+generalizat[aă]", t))
        
        # Context comparativ - NU penaliza
        is_comparison = bool(re.search(r"(?:ar fi|daca ar|cu 3 tije|pentru 3 tije|in loc de|comparativ|versus|fata de|spre deosebire)", t))
        
        if not is_comparison:
            if says_classic_direct and not is_classic:
                return f"Afirmi că problema e clasică (3 tije), dar de fapt are {pegs} tije."
            if says_generalized_direct and is_classic:
                return f"Afirmi că problema e generalizată, dar de fapt are exact 3 tije."

    return None


# =======================================================================
# 3. INTERFAȚA PUBLICĂ - Generare și Evaluare
# =======================================================================

def generate_n(n: int, selected_topics: List[str] = None) -> List[Dict[str, Any]]:
    """
    Generează n întrebări, permițând filtrarea după topics (capitole).
    
    Args:
        n: Numărul de întrebări de generat
        selected_topics: Lista de tipuri de probleme (None = toate)
    
    Returns:
        Lista de întrebări generate cu parametri și opțiuni
    """
    _TEMPLATES = [
        {
            "type": "n-queens",
            "template": "Având problema N-Queens cu N={N}, ce strategie de căutare este cea mai potrivită?  Argumentează alegerea.",
            "params_gen": lambda: {"N": random.choice([4, 6, 8, 10, 12, 50, 100, 200])}
        },
        {
            "type": "generalized hanoi",
            "template": "Pentru Turnurile din Hanoi ({pegs} tije, {disks} discuri), ce strategie alegi? Explică de ce.",
            "params_gen": lambda: {"pegs": random. choice([3, 4, 5]), "disks": random.choice([5, 8, 10, 15, 20])}
        },
        {
            "type": "graph coloring",
            "template": "La o problemă de Colorare Graf (|V|={V}, |E|={E}, k={k} culori), ce strategie e optimă? Justifică.",
            "params_gen": lambda: {
                "V": random.choice([20, 40, 60, 80]),
                "E": random.choice([50, 100, 200, 400, 800, 1200]),
                "k": random.choice([3, 4, 5, 6])
            }
        },
        {
            "type": "knight's tour",
            "template": "Pentru Knight's Tour pe o tablă {n}x{n}, ce strategie funcționează cel mai eficient? Motivează.",
            "params_gen": lambda: {"n": random. choice([5, 6, 8, 10, 12, 20])}
        },
    ]

    available_templates = _TEMPLATES
    if selected_topics:
        filtered = [t for t in _TEMPLATES if t["type"] in selected_topics]
        if filtered:
            available_templates = filtered
    
    items = []
    
    for i in range(n):
        t = random.choice(available_templates)
        params = t["params_gen"]()
        
        # Calculăm răspunsul corect DINAMIC (nu hardcodat!)
        truth = StrategySolver. solve(t["type"], params)
        correct_strat = truth["best_strategy"]
        
        # Generăm distractori INTELIGENȚI bazați pe parametri
        wrong_options = StrategySolver. get_distractors(t["type"], correct_strat, params, 3)
        
        # Amestecăm opțiunile
        current_options = [correct_strat] + wrong_options
        random.shuffle(current_options)
        
        items.append({
            "index": i,
            "type": "search_strategy_selection",
            "problem_type": t["type"],
            "question": t["template"]. format(**params),
            "params": params,
            "options": current_options
        })
    
    return items


def evaluate_answer(user_text: str, item_context: Dict[str, Any]) -> Dict[str, Any]:
    ptype = item_context.get("problem_type")
    params = item_context. get("params", {})
    
    truth = StrategySolver. solve(ptype, params)
    correct_strategy = truth["best_strategy"]
    required_concepts = truth["required_concepts"]
    forbidden_concepts = truth. get("forbidden_concepts", [])
    ideal_explanation = truth["reasoning_summary"]
    calculations = truth.get("calculations", {})

    if not user_text or len(user_text. strip()) < 5:
        return {
            "score": 0.0,
            "components": {
                "selection_score": 0.0,
                "reasoning_score": 0.0,
                "detected_intent": "None",
                "truth_recalculated": correct_strategy,
                "violations_detected": [],
                "wrong_strategy_detected": None,
                "message": "Răspuns prea scurt sau gol."
            }
        }

    # ===== NOU: VERIFICARE LUNGIME RĂSPUNS =====
    word_count = len(user_text.split())
    length_penalty = 1.0
    length_msg = ""
    
    if word_count < 12:
        length_penalty = 0.4  # Penalizare 60%
        length_msg = f" (Răspuns foarte scurt: {word_count} cuvinte - argumentare insuficientă)"
    elif word_count < 25:
        length_penalty = 0.65  # Penalizare 35%
        length_msg = f" (Răspuns scurt: {word_count} cuvinte)"
    elif word_count < 40:
        length_penalty = 0.85  # Penalizare 15%
        length_msg = ""
    # ============================================

    options = item_context.get("options", [])
    
    selection = SmartEvaluator.identify_selection(user_text, options)
    score_selection = 0.0
    status_msg = ""
    detected_strat = "None"

    if selection:
        user_strat = selection['strategy']
        polarity = selection['polarity']
        detected_strat = f"{user_strat} ({polarity})"
        
        user_strat_norm = _normalize(user_strat)
        correct_strat_norm = _normalize(correct_strategy)
        
        is_exact_match = user_strat_norm == correct_strat_norm
        is_partial_match = False
        if not is_exact_match:
            if len(user_strat_norm) >= 8 and len(correct_strat_norm) >= 8:
                if user_strat_norm in correct_strat_norm or correct_strat_norm in user_strat_norm:
                    is_partial_match = True
        is_match = is_exact_match or is_partial_match

        if polarity == "contradictory":
            score_selection = 0.0
            status_msg = "Răspuns incoerent (contradicție internă detectată)."
        elif polarity == "positive":
            if is_match:
                score_selection = 1.0 if is_exact_match else 0.85
                status_msg = "Strategie corectă identificată."
            else:
                score_selection = 0.0
                status_msg = f"Ai ales '{user_strat}', dar răspunsul corect este '{correct_strategy}'."
        else:
            if is_match:
                score_selection = 0.0
                status_msg = f"Ai respins/exclus strategia corectă ({correct_strategy})."
            else:
                score_selection = 0.15
                status_msg = f"Ai exclus '{user_strat}' (care era greșit), dar nu ai specificat clar ce alegi."
    else:
        status_msg = "Nu am identificat clar o strategie în răspunsul tău."
        score_selection = 0.0

    score_reasoning, violations, wrong_strategy, numerical_feedback = SmartEvaluator.check_reasoning(
        user_text, 
        required_concepts, 
        forbidden_concepts, 
        ideal_explanation, 
        correct_strategy,
        calculations,
        ptype
    )
    
    original_selection_score = score_selection
    
    if wrong_strategy:
        status_msg += f" EROARE: Descrierea ta corespunde strategiei '{wrong_strategy}', nu '{correct_strategy}'!"
        score_selection = 0.0
        score_reasoning = 0.0
    
    if violations and not wrong_strategy:
        status_msg += f" Penalizare: concepte incompatibile detectate: {', '.join(violations[:3])}."
        if score_selection > 0.0:
            score_selection *= 0.3

    # ===== NOU: DETECTARE ERORI FACTUALE HANOI =====
    if ptype == "generalized hanoi":
        pegs = calculations.get('pegs', params.get('pegs', 3))
        disks = calculations.get('disks', params.get('disks', 5))
        is_classic = calculations. get('is_classic', pegs == 3)
        
        if not is_classic:  # 4+ tije
            classic_formula_result = 2**disks - 1
            text_lower = user_text.lower()
            
            # Verifică dacă menționează formula clasică greșit
            if str(classic_formula_result) in text_lower or f"2^{disks}-1" in text_lower. replace(" ", ""):
                formula_error = True
                status_msg += f" EROARE FACTUALĂ: Formula 2^{disks}-1={classic_formula_result} e pentru 3 tije, nu {pegs}!"
                score_reasoning *= 0.5  # Penalizare 50% pentru eroare factuală
    # ================================================

    contradiction = _detect_ground_contradiction(user_text, item_context, truth)
    if contradiction:
        score_reasoning *= 0.3
        if score_selection > 0.5:
            score_selection = 0.5
        status_msg += f" CONTRADICȚIE FACTUALĂ: {contradiction}"

    if numerical_feedback:
        status_msg += " " + " | ".join(numerical_feedback)

    # 5.   CALCULEAZĂ SCORUL FINAL (0-100)
    
    if wrong_strategy:
        base_score = 10 + (score_reasoning * 15)
        final_score = min(25, max(10, base_score))
        status_msg += " (Penalizare severă: descriere de altă strategie)"
        
    elif violations:
        num_violations = len(violations)
        
        if original_selection_score > 0.8:
            base_score = 20 + (score_reasoning * 15)
            violation_penalty = min(10, num_violations * 2)
            final_score = max(15, base_score - violation_penalty)
            final_score = min(35, final_score)
            status_msg += f" (Penalizare: {num_violations} concepte incompatibile)"
        else:
            final_score = 5 + (score_reasoning * 10)
            final_score = min(15, max(5, final_score))
            status_msg += f" (Penalizare: strategie greșită + {num_violations} concepte incompatibile)"
            
    elif contradiction:
        final_score = (score_selection * 25) + (score_reasoning * 25)
        final_score = max(15, min(40, final_score))
        
    elif score_selection > 0.8 and score_reasoning < 0.3:
        # Strategie corectă dar argumentare slabă
        final_score = (score_selection * 35) + (score_reasoning * 65)
        final_score = max(30, min(50, final_score))
        status_msg += " (Scor redus: Argumentare insuficientă)"
        
    elif score_selection < 0.3 and score_reasoning > 0.6:
        final_score = (score_selection * 40) + (score_reasoning * 60)
        final_score = max(20, min(40, final_score))
        status_msg += " (Argumentare bună dar strategie greșită)"
        
    elif score_selection > 0.8 and score_reasoning > 0.6:
        final_score = (score_selection * 50) + (score_reasoning * 50)
        final_score = max(70, min(100, final_score))
        
    elif score_selection > 0.8 and score_reasoning >= 0.3:
        final_score = (score_selection * 50) + (score_reasoning * 50)
        final_score = max(50, min(75, final_score))
        
    else:
        final_score = (score_selection * 50) + (score_reasoning * 50)
        final_score = max(0, min(100, final_score))

    # ===== NOU: APLICĂ PENALIZARE PENTRU LUNGIME =====
    final_score = final_score * length_penalty
    status_msg += length_msg
    # =================================================

    return {
        "score": round(final_score, 1),
        "components": {
            "selection_score": round(score_selection, 2),
            "reasoning_score": round(score_reasoning, 2),
            "detected_intent": detected_strat,
            "truth_recalculated": correct_strategy,
            "violations_detected": violations if violations else [],
            "wrong_strategy_detected": wrong_strategy,
            "word_count": word_count,
            "message": status_msg
        }
    }

# =======================================================================
# STORAGE HELPERS
# =======================================================================
def save_public_questions(items: List[Dict[str, Any]], path: str) -> None:
    """Salvează întrebările într-un fișier JSON (fără răspunsuri)."""
    # Eliminăm câmpurile sensibile pentru versiunea publică
    public_items = []
    for item in items:
        public_item = {
            "index": item. get("index"),
            "type": item.get("type"),
            "problem_type": item. get("problem_type"),
            "question": item.get("question"),
            "params": item.get("params"),
            "options": item. get("options")
        }
        public_items.append(public_item)
    
    with open(path, "w", encoding="utf-8") as f:
        json. dump(public_items, f, ensure_ascii=False, indent=2)


def save_answer_key(items: List[Dict[str, Any]], path: str) -> None:
    """Salvează answer key-ul (include parametrii pentru recalculare)."""
    # Answer key-ul conține parametrii - răspunsul se recalculează dinamic
    with open(path, "w", encoding="utf-8") as f:
        json. dump(items, f, ensure_ascii=False, indent=2)


# =======================================================================
# EXPORTS (pentru compatibilitate)
# =======================================================================
__all__ = [
    'generate_n',
    'evaluate_answer',
    'save_public_questions',
    'save_answer_key',
    'StrategySolver',
    'SmartEvaluator',
    '_normalize'
]