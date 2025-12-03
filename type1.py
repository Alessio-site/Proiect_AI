import random
import re
import json
import spacy
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher

# =======================================================================
# 0. SETUP NLP & AI MODELS (Local)
# =======================================================================

# 1. SpaCy pentru analiză sintactică rapidă și negații (Lightweight)
try:
    nlp = spacy.load("ro_core_news_lg")
except OSError:
    print("Modelul 'ro_core_news_lg' nu este instalat. Se încearcă 'ro_core_news_sm'...")
    try:
        nlp = spacy.load("ro_core_news_sm")
    except Exception:
        print("Eroare critică: Niciun model spaCy de română găsit. Evaluarea va fi limitată.")
        nlp = None

# 2. Sentence Transformers pentru evaluare semantică profundă (Conform Raport Cercetare)
# Aceasta înlocuiește logica simplistă de similaritate cu una bazată pe Transformers (SBERT)
semantic_model = None
try:
    from sentence_transformers import SentenceTransformer, util
    print("Se încarcă modelul semantic SBERT (poate dura puțin la prima rulare)...")
    # Model multilingv compact și rapid, ideal pentru CPU
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except ImportError:
    print("WARN: `sentence-transformers` nu este instalat. Se va folosi fallback pe cuvinte cheie.")
except Exception as e:
    print(f"WARN: Eroare la încărcarea SBERT: {e}")

# =======================================================================
# Helpers: normalizare & lexicon extins pentru negare
# =======================================================================
def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, remove diacritics and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\"'()\[\]{}:;,. !?\\/<>@#%^&*+=~`|\\]+", " ", text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
    "ar", "s-ar", "fi", "decat", "fata", "versus", "comparativ", "instead", "would", "could", "should", "fata de", "spre deosebire"
}

REJECTION_PATTERNS = [
    r"nu\s+(?:vom\s+)?(?:folosi|alege|opta|recomanda)\s+(.+?)(?:\s+|$|,|\.)",
    r"(?:evitam|excludem|respingem|refuzam)\s+(.+?)(?:\s+|$|,|\.)",
    r"(.+?)\s+(?:nu\s+(?:este|e)|ar\s+fi\s+gresit|nu\s+functioneaza)",
    r"(?:in\s+loc\s+de|spre\s+deosebire\s+de)\s+(.+?)(?:\s+|,|\.)",
]

# =======================================================================
# Dicționar de "semnături" pentru fiecare strategie
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

def _detect_strategy_from_description(text: str) -> Optional[str]:
    text_norm = _normalize(text)
    scores = {}
    for strategy, patterns in STRATEGY_SIGNATURES.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_norm):
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
# 1. LOGICĂ PURĂ (Solver-ul Matematic)
# =======================================================================
class StrategySolver:
    # NOU: Definim distractorii relevanți (opțiuni greșite plauzibile)
    # pentru a face întrebările grilă mai inteligente.
    RELEVANT_DISTRACTORS = {
        "n-queens": [
            "plain dfs", "plain bfs", "genetic algorithm", "simulated annealing", "a*"
        ],
        "generalized hanoi": [
            "plain bfs", "plain dfs", "ida*", "dynamic programming", "min-conflicts heuristic"
        ],
        "graph coloring": [
            "greedy coloring", "plain dfs", "genetic algorithm", "simulated annealing", "plain bfs"
        ],
        "knight's tour": [
            "backtracking", "plain bfs", "plain dfs", "simulated annealing"
        ]
    }

    @staticmethod
    def get_distractors(problem_type: str, correct_strategy: str, count: int = 3) -> List[str]:
        """Selectează distractori relevanți pentru tipul problemei."""
        candidates = StrategySolver.RELEVANT_DISTRACTORS.get(problem_type, [])
        # Eliminăm varianta corectă dacă apare din greșeală în distractori
        candidates = [c for c in candidates if _normalize(c) != _normalize(correct_strategy)]
        
        # Dacă nu avem destui candidați specifici, completăm cu alții generici
        if len(candidates) < count:
            all_strats = list(STRATEGY_SIGNATURES.keys())
            others = [s for s in all_strats if s not in candidates and _normalize(s) != _normalize(correct_strategy)]
            random.shuffle(others)
            candidates.extend(others[:count - len(candidates)])
            
        return random.sample(candidates, min(len(candidates), count))

    @staticmethod
    def solve(problem_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculează matematic care este strategia corectă pe baza parametrilor.
        Aceasta asigură un 'răspuns corect garantat'.
        """
        truth = {
            "best_strategy": None, 
            "required_concepts": [], 
            "forbidden_concepts": [],
            "reasoning_summary": ""
        }

        if problem_type == "n-queens":
            N = int(params.get("N", 8))
            if N < 15:
                truth["best_strategy"] = "backtracking with MRV + forward checking"
                truth["required_concepts"] = ["complet", "mic", "manageable", "sigur", "toate", "mrv", "spatiu", "garanteaza"]
                truth["forbidden_concepts"] = [
                    "local", "conflicte", "hill", "gradient", "temperatura", "rapid", "probabilistic",
                    "energie", "racire", "cooling", "populatie", "genetic", "fitness", "evolutie",
                    "annealing", "configuratie aleatoare", "accepta mutari"
                ]
                truth["reasoning_summary"] = f"Pentru N={N}, spațiul este mic și gestionabil. Backtracking cu MRV este o strategie completă și sigură, garantând găsirea soluției."
            else:
                truth["best_strategy"] = "min-conflicts heuristic"
                truth["required_concepts"] = ["rapid", "local", "conflicte", "mare", "timp", "eficient", "euristica", "scala"]
                truth["forbidden_concepts"] = [
                    "complet", "exhaustiv", "toate", "arbore", "memorie", "sigur",
                    "temperatura", "racire", "energie", "populatie", "genetic", "crossover"
                ]
                truth["reasoning_summary"] = f"Pentru N={N}, spațiul este enorm. Algoritmii compleți (Backtracking) sunt prea lenți. Min-conflicts este o metodă de căutare locală mult mai rapidă."

        elif problem_type == "generalized hanoi":
            truth["best_strategy"] = "recursive strategy (divide and conquer)"
            truth["required_concepts"] = ["recursiv", "structura", "natural", "subprobleme", "optim", "divide", "conquer", "exponential"]
            truth["forbidden_concepts"] = [
                "ac-3", "arc", "consistenta", "constrangeri", "propagare", "domeniu",
                "conflicte", "min-conflicts", "local", "vecini", "swap",
                "temperatura", "racire", "racindu", "energie", "cooling", "annealing",
                "configuratie aleatoare", "accepta mutari", "probabilitate", "treptat",
                "gradual", "metropolis", "boltzmann", "cresc energia",
                "populatie", "selectie", "crossover", "mutatie", "fitness", "generatii",
                "evolutie", "cromozom", "gene", "parinti",
                "hill", "climbing", "gradient",
                "exhaustiv", "nivel cu nivel", "coada", "stiva"
            ]
            truth["reasoning_summary"] = "Problema are o structură recursivă naturală. Divide et Conquer oferă soluția optimă (număr minim de mutări) fără a explora stări inutile."

        elif problem_type == "graph coloring":
            V = int(params.get("V", 10))
            E = int(params.get("E", 10))
            max_edges = V * (V - 1) / 2 if V > 1 else 1
            density = E / max_edges

            if density > 0.35:
                truth["best_strategy"] = "ac-3 with backtracking"
                truth["required_concepts"] = ["dens", "constrangeri", "arc", "consistenta", "propagare", "ac-3", "prune"]
                truth["forbidden_concepts"] = [
                    "rar", "sparse", "simplu", "min-conflicts", "genetic", "recursiv",
                    "temperatura", "racire", "energie", "populatie", "fitness"
                ]
                truth["reasoning_summary"] = f"Graful este dens (densitate {density:.2f}). Constrângerile sunt multe, deci AC-3 este necesar pentru a reduce domeniile înainte de căutare."
            else:
                truth["best_strategy"] = "backtracking with MRV + forward checking"
                truth["required_concepts"] = ["rar", "sparse", "mrv", "rapid", "simplu", "forward", "checking", "csp"]
                truth["forbidden_concepts"] = [
                    "dens", "ac-3", "arc", "consistenta", "recursiv", "temperatura",
                    "energie", "racire", "populatie", "genetic", "fitness"
                ]
                truth["reasoning_summary"] = f"Graful este rar (sparse, densitate {density:.2f}). MRV și Forward Checking sunt suficiente și au un overhead mai mic decât AC-3."

        elif problem_type == "knight's tour":
            truth["best_strategy"] = "warnsdorff heuristic (greedy)"
            truth["required_concepts"] = ["greedy", "regula", "rapid", "liniar", "grad", "mutari", "euristica", "warnsdorff"]
            truth["forbidden_concepts"] = [
                "bfs", "dfs", "complet", "exhaustiv", "backtracking", "recursiv", "genetic",
                "temperatura", "racire", "energie", "populatie", "fitness", "conflicte"
            ]
            truth["reasoning_summary"] = "Euristica Warnsdorff (alege nodul cu gradul minim) este extrem de rapidă și găsește soluția în timp liniar în majoritatea cazurilor, spre deosebire de Backtracking."

        return truth

    @staticmethod
    def get_system_answer(problem_type: str, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Generează textul răspunsului pe care îl dă aplicația însăși.
        Îndeplinește cerința: 'Răspunsul poate fi punctual sau detaliat'.
        """
        solution = StrategySolver.solve(problem_type, params)
        strategy = solution["best_strategy"]
        reasoning = solution["reasoning_summary"]
        
        detailed_text = (
            f"Strategia optimă este: {strategy}.\n\n"
            f"Explicație și Argumentare:\n{reasoning}\n"
            f"Analizând parametrii instanței {json.dumps(params)}, această strategie oferă cel mai bun echilibru între timp de execuție și completitudine."
        )

        return {
            "punctual": strategy,
            "detailed": detailed_text
        }

# =======================================================================
# 2. NLP ENGINE (Analizatorul de Intenție & Evaluare Semantică)
# =======================================================================
class SmartEvaluator:
    # Pattern-uri Regex pentru identificarea strategiilor
    STRATEGY_PATTERNS = [
        (r"backtracking\s+(?:with\s+)?mrv\s*\+?\s*forward\s*checking", "backtracking with mrv + forward checking"),
        (r"backtracking\s+(?:cu\s+)?mrv\s+(?:si|și)\s+forward\s*checking", "backtracking with mrv + forward checking"),
        (r"mrv\s*\+?\s*forward\s*checking", "backtracking with mrv + forward checking"),
        (r"recursive\s+strategy\s*\(?\s*divide\s+(?:and|et|si|și)\s+conquer\s*\)?", "recursive strategy (divide and conquer)"),
        (r"recursive\s+strategy", "recursive strategy (divide and conquer)"),
        (r"divide\s+(?:and|et|si|și)\s+conquer", "recursive strategy (divide and conquer)"),
        (r"warnsdorff", "warnsdorff heuristic (greedy)"),
        (r"ac-?3\s+(?:with\s+)?backtracking", "ac-3 with backtracking"),
        (r"arc\s+consistency", "ac-3 with backtracking"),
        (r"min-?conflicts?", "min-conflicts heuristic"),
        (r"genetic\s+algorithm", "genetic algorithm"),
        (r"simulated\s+annealing", "simulated annealing"),
        (r"hill\s+climbing", "hill climbing with sideways moves"),
        (r"\bplain\s+bfs\b", "plain bfs"),
        (r"\bbreadth[- ]first\b", "plain bfs"),
        (r"\bplain\s+dfs\b", "plain dfs"),
        (r"\bdepth[- ]first\b", "plain dfs"),
        (r"\ba\*\b", "a*"),
        (r"\bida\*\b", "ida*"),
        (r"\bdynamic\s+programming\b", "dynamic programming"),
        (r"\bbacktracking\b(?!\s+with)", "backtracking with mrv + forward checking"),
        (r"\bac-?3\b", "ac-3 with backtracking"),
    ]

    @staticmethod
    def _find_all_strategies(text: str) -> List[Dict[str, Any]]:
        text_norm = _normalize(text)
        found = []
        used = []
        for pattern, canonical in SmartEvaluator.STRATEGY_PATTERNS:
            for m in re.finditer(pattern, text_norm):
                s, e = m.start(), m.end()
                overlap = any(not (e <= us or s >= ue) for us, ue in used)
                if not overlap:
                    used.append((s, e))
                    found.append({'canonical': canonical, 'start': s, 'end': e, 'matched_text': m.group()})
        found.sort(key=lambda x: x['start'])
        return found

    @staticmethod
    def _detect_criticism_after_mention(text: str, mention: Dict[str, Any]) -> bool:
        text_norm = _normalize(text)
        end_pos = mention['end']
        after_text = text_norm[end_pos:end_pos + 200]
        
        # FIX: Am adăugat \b (word boundary) pentru a evita potriviri false 
        # (ex: "nu" în "numarul", "dar" în "daruieste")
        criticism_patterns = [
            r"\b(?:dar|insa|totusi|however|but)\b\s+.{0,50}\b(?:ignora|opreste|blocheaza|nu\s+(?:functioneaza|merge)|lipseste|problema)\b",
            r"(?:implementarea|aceasta|aceasta\s+metoda)\s+.{0,30}\b(?:ignora|nu|lipseste|opreste)\b",
            r"(?:mrv|forward\s+checking)\s+.{0,30}\b(?:ignora|nu|lipseste|opreste|ineficient)\b",
            r"\bin\s+consecinta\s+(?:aplicam|folosim|alegem)\b",
            r"(?:de\s+aceea|astfel|prin\s+urmare)\s+(?:aplicam|folosim|alegem)\b",
        ]
        for pattern in criticism_patterns:
            if re.search(pattern, after_text):
                return True
        return False

    @staticmethod
    def _detect_alternative_proposal(text: str, primary_mention: Dict[str, Any], all_mentions: List[Dict[str, Any]]) -> Optional[str]:
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
    def _analyze_answer_coherence(text: str, selected_strategy: str, all_mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_norm = _normalize(text)
        selected_norm = _normalize(selected_strategy)
        result = {'is_coherent': True, 'criticism_detected': False, 'alternative_proposed': None, 'penalty_reason': None}
        
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
        text_norm = _normalize(text)
        start = mention['start']
        end = mention['end']
        
        left_window = text_norm[max(0, start - 90):start]
        for pat in REJECTION_PATTERNS:
            if re.search(pat, left_window): return 'negative'

        immediate_left = text_norm[max(0, start - 30):start]
        if any(tok in NEGATION_WORDS for tok in immediate_left.split()): return 'negative'
        if any(neg in left_window for neg in NEGATION_VERBS): return 'negative'

        immediate_right = text_norm[end:end + 30]
        if any(cond in immediate_right.split() for cond in CONDITIONAL_WORDS):
            return 'negative'

        if nlp_doc and nlp:
            try:
                for token in nlp_doc:
                    if token.idx <= mention['start'] * 1.2:
                        if any(c.dep_ == 'neg' or _normalize(c.text) in NEGATION_WORDS for c in token.children): return 'negative'
                        if token.head and token.head.pos_ == 'VERB':
                            if _normalize(token.head.lemma_) in NEGATION_VERBS: return 'negative'
            except Exception: pass
        return 'positive'

    @staticmethod
    def identify_selection(text: str, context_options: List[str]) -> Optional[Dict[str, Any]]:
        if not text or len(text.strip()) < 3: return None
        context_norm = [_normalize(o) for o in context_options]
        mentions = SmartEvaluator._find_all_strategies(text)
        if not mentions: return None
        
        nlp_doc = None
        if nlp:
            try: nlp_doc = nlp(text)
            except Exception: nlp_doc = None
        
        analyzed = []
        for m in mentions:
            polarity = SmartEvaluator._detect_polarity_for_mention(text, m, nlp_doc)
            analyzed.append({**m, 'polarity': polarity, 'in_context': any(m['canonical'].lower() in c or c in m['canonical'].lower() for c in context_norm)})
        
        pos_in_ctx = [a for a in analyzed if a['polarity'] == 'positive' and a['in_context']]
        if pos_in_ctx:
            best = pos_in_ctx[0]
            coherence = SmartEvaluator._analyze_answer_coherence(text, best['canonical'], mentions)
            return {'strategy': best['canonical'], 'polarity': 'positive' if coherence['is_coherent'] else 'contradictory', 'confidence': 0.95 if coherence['is_coherent'] else 0.3, 'coherence_issue': coherence['penalty_reason'], 'alternative_proposed': coherence['alternative_proposed']}
        
        pos_any = [a for a in analyzed if a['polarity'] == 'positive']
        if pos_any:
            best = pos_any[0]
            coherence = SmartEvaluator._analyze_answer_coherence(text, best['canonical'], mentions)
            return {'strategy': best['canonical'], 'polarity': 'positive' if coherence['is_coherent'] else 'contradictory', 'confidence': 0.8 if coherence['is_coherent'] else 0.3, 'coherence_issue': coherence['penalty_reason'], 'alternative_proposed': coherence['alternative_proposed']}
        
        if analyzed:
            best = analyzed[0]
            return {'strategy': best['canonical'], 'polarity': 'negative', 'confidence': 0.7, 'coherence_issue': None, 'alternative_proposed': None}
        return None

    @staticmethod
    def check_reasoning(user_text: str, required_concepts: List[str], forbidden_concepts: List[str], ideal_explanation: str = "", correct_strategy: str = "") -> Tuple[float, List[str], Optional[str]]:
        """
        Analizează argumentarea folosind atât detecția de concepte (Regex/SpaCy),
        cât și similaritatea semantică avansată (SBERT), conform cerințelor din Raport.
        """
        if not user_text or not user_text.strip(): return 0.0, [], None
        text_norm = _normalize(user_text)
        
        # 1. DETECTARE STRATEGIE DIN DESCRIERE
        described_strategy = _detect_strategy_from_description(user_text)
        wrong_strategy_detected = None
        if described_strategy and correct_strategy:
            correct_norm = _normalize(correct_strategy)
            described_norm = _normalize(described_strategy)
            if described_norm != correct_norm and correct_norm not in described_norm and described_norm not in correct_norm:
                wrong_strategy_detected = described_strategy
        
        # 2. VERIFICARE CONCEPTE TOXICE
        violations = []
        for bad in forbidden_concepts:
            bad_norm = _normalize(bad)
            pattern = r"(.{0,50})\b" + re.escape(bad_norm) + r"\b(.{0,50})"
            m = re.search(pattern, text_norm)
            if m:
                left = m.group(1)
                full_context = m.group(0)
                is_safe = False
                if any(w in left for w in NEGATION_WORDS) or any(w in left for w in NEGATION_VERBS): is_safe = True
                if any(w in full_context.split() for w in CONDITIONAL_WORDS): is_safe = True
                if not is_safe:
                    violations.append(bad)

        if violations or wrong_strategy_detected:
            return 0.1, violations, wrong_strategy_detected

        # 3. VERIFICARE CONCEPTE NECESARE (Bazat pe Regex/Spacy)
        hits = 0
        doc = None
        if nlp:
            try: doc = nlp(user_text)
            except Exception: pass
        
        if doc:
            for concept in required_concepts:
                c_norm = _normalize(concept)
                found_pos = False
                for token in doc:
                    tok_norm = _normalize(token.text)
                    if c_norm in tok_norm:
                        # Context Check
                        is_neg = False
                        left_tokens = list(doc[max(0, token.i - 4):token.i])
                        if any(_normalize(t.text) in NEGATION_WORDS for t in left_tokens): is_neg = True
                        if not is_neg: found_pos = True; break
                if found_pos: hits += 1
            keyword_score = hits / max(1, len(required_concepts)) if required_concepts else 1.0
        else:
            keyword_score = sum(1 for c in required_concepts if _normalize(c) in text_norm) / max(1, len(required_concepts))

        # 4. SEMANTIC SIMILARITY UPGRADED (SBERT) - Conform Raport
        similarity_score = 0.0
        if ideal_explanation and semantic_model:
            try:
                # Calculăm vectorii (Embeddings)
                emb_ideal = semantic_model.encode(ideal_explanation, convert_to_tensor=True)
                emb_user = semantic_model.encode(user_text, convert_to_tensor=True)
                
                # Calculăm similaritatea Cosinus
                similarity_score = util.cos_sim(emb_ideal, emb_user).item()
                
                # Normalizare: SBERT tinde să dea scoruri >0.3. Scalăm [0.3, 0.9] -> [0.0, 1.0]
                similarity_score = max(0.0, (similarity_score - 0.3) / 0.6)
                similarity_score = min(1.0, similarity_score)
            except Exception as e:
                print(f"[ERR] SBERT error: {e}")
                similarity_score = 0.0
        
        # Pondere finală: 40% Cuvinte Cheie + 60% Semantică Profundă
        final_score = (keyword_score * 0.4) + (similarity_score * 0.6)
        
        return min(1.0, final_score), [], None

# =======================================================================
# Utility pentru detectare contradicții explicite în text
# =======================================================================
def _detect_ground_contradiction(user_text: str, item_context: Dict[str, Any], truth: Dict[str, Any]) -> Optional[str]:
    t = _normalize(user_text)
    ptype = item_context.get('problem_type')
    params = item_context.get('params', {})

    if ptype == 'graph coloring':
        V = int(params.get('V', 0))
        E = int(params.get('E', 0))
        max_edges = V * (V - 1) / 2 if V > 1 else 1
        density = E / max_edges
        says_dense = bool(re.search(r"\bdens\b|\bdensitate\b", t))
        says_not_dense = bool(re.search(r"\bnu\s+este\s+dens\b|\bnu\s+dens\b", t))
        if says_not_dense and density > 0.35: return f"User afirmă că graful NU este dens, dar densitatea reală este {density:.2f}."
        if says_dense and density <= 0.35: return f"User afirmă că graful este dens, dar densitatea reală este {density:.2f} (rar)."

    if ptype == 'n-queens':
        N = int(params.get('N', 8))
        says_small = bool(re.search(r"\bmic\b", t))
        says_large = bool(re.search(r"\bmare\b", t))
        if says_small and N >= 15: return f"User afirmă că N e mic, dar N={N} (mare)."
        if says_large and N < 15: return f"User afirmă că N e mare, dar N={N} (mic)."

    return None

# =======================================================================
# 3. INTERFAȚA PUBLICĂ
# =======================================================================
def generate_n(n: int, selected_topics: List[str] = None) -> List[Dict[str, Any]]:
    """
    Generează n întrebări, permițând filtrarea după topics (capitole).
    """
    _TEMPLATES = [
        {"type": "n-queens", "template": "Având problema N-Queens cu N={N}, ce strategie de căutare este cea mai potrivită? Argumentează.", "params_gen": lambda: {"N": random.choice([8, 10, 50, 100])}},
        {"type": "generalized hanoi", "template": "Pentru Turnurile din Hanoi ({pegs} tije, {disks} discuri), ce strategie alegi? Explica de ce.", "params_gen": lambda: {"pegs": 3, "disks": random.choice([6, 10, 15])}},
        {"type": "graph coloring", "template": "La o problemă de Colorare Graf (|V|={V}, |E|={E}, k={k}), ce strategie e optimă? Justifica.", "params_gen": lambda: {"V": 60, "E": random.choice([80, 800]), "k": 4}},
        {"type": "knight's tour", "template": "Pentru Knight's Tour pe o tablă {n}x{n}, ce strategie funcționează cel mai eficient? Motiv.", "params_gen": lambda: {"n": random.choice([8, 20])}},
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
        
        # Asigură-te că există un răspuns corect garantat (calculat dinamic)
        truth = StrategySolver.solve(t["type"], params)
        correct_strat = truth["best_strategy"]
        
        # AICI ALEGEM DISTRACTORI RELEVANȚI, NU DOAR RANDOM
        wrong_options = StrategySolver.get_distractors(t["type"], correct_strat, 3)
        current_options = [correct_strat] + wrong_options
        random.shuffle(current_options)
        
        items.append({
            "index": i, 
            "type": "search_strategy_selection", 
            "problem_type": t["type"], 
            "question": t["template"].format(**params), 
            "params": params, 
            "options": current_options
        })
    return items

def evaluate_answer(user_text: str, item_context: Dict[str, Any]) -> Dict[str, Any]:
    ptype = item_context.get("problem_type")
    params = item_context.get("params", {})
    truth = StrategySolver.solve(ptype, params)
    correct_strategy = truth["best_strategy"]
    required_concepts = truth["required_concepts"]
    forbidden_concepts = truth.get("forbidden_concepts", [])
    ideal_explanation = truth["reasoning_summary"]

    if not user_text or len(user_text.strip()) < 5:
        return {"score": 0.0, "components": {"selection_score": 0.0, "reasoning_score": 0.0, "detected_intent": "None", "truth_recalculated": correct_strategy, "message": "Răspuns prea scurt sau gol."}}

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
            status_msg = "Răspuns incoerent (contradicție detectată)."
        elif polarity == "positive":
            if is_match:
                score_selection = 1.0 if is_exact_match else 0.85
                status_msg = "Strategie corectă identificată."
            else:
                score_selection = 0.0
                status_msg = f"Ai ales {user_strat}, dar trebuia {correct_strategy}."
        else: # negative
            if is_match:
                score_selection = 0.0
                status_msg = f"Ai respins/exclus strategia corectă ({correct_strategy})."
            else:
                score_selection = 0.15
                status_msg = f"Ai exclus {user_strat} (care era greșit), dar nu ai specificat ce alegi."
    else:
        status_msg = "Nu am identificat clar o strategie în răspunsul tău."
        score_selection = 0.0

    score_reasoning, violations, wrong_strategy = SmartEvaluator.check_reasoning(
        user_text, required_concepts, forbidden_concepts, ideal_explanation, correct_strategy
    )
    
    if wrong_strategy:
        status_msg += f" EROARE: Descrierea ta corespunde strategiei '{wrong_strategy}', nu '{correct_strategy}'!"
        score_selection = 0.0 
        score_reasoning = 0.0 
    
    if violations:
        status_msg += f" Penalizare: concepte greșite: {', '.join(violations)}."
        if score_selection > 0.0:
            score_selection *= 0.3 

    contradiction = _detect_ground_contradiction(user_text, item_context, truth)
    if contradiction:
        score_reasoning = 0.0
        if score_selection > 0.5:
            score_selection = 0.5
        status_msg += f" CONTRADICȚIE: {contradiction}"

    # Pondere finală
    if wrong_strategy or violations:
        final_score = (score_selection * 20) + (score_reasoning * 10)
    elif score_selection > 0.8 and score_reasoning < 0.3:
        final_score = (score_selection * 30) + (score_reasoning * 70)
        status_msg += " (Scor redus: Argumentare insuficientă)"
    else:
        final_score = (score_selection * 50) + (score_reasoning * 50)

    return {
        "score": round(final_score, 1),
        "components": {
            "selection_score": round(score_selection, 2),
            "reasoning_score": round(score_reasoning, 2),
            "detected_intent": detected_strat,
            "truth_recalculated": correct_strategy,
            "message": status_msg
        }
    }

# =======================================================================
# STORAGE HELPERS
# =======================================================================
def save_public_questions(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_answer_key(items: List[Dict[str, Any]], path: str) -> None:
    save_public_questions(items, path)