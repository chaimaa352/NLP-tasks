# NLP : Résumé automatique, Question-Réponse et Génération de texte

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaimaa352/NLP-tasks/blob/main/nlp_tasks.ipynb)

Documentation du code source du notebook `nlp_tasks.ipynb`.

Le notebook implémente trois tâches du traitement automatique du langage naturel (NLP) avec des modèles **Hugging Face exécutés entièrement en local** (aucune API cloud), les évalue avec des métriques standard, puis **analyse** les résultats (erreurs, robustesse, vitesse).

| # | Tâche | Modèle | Métriques |
|---|-------|--------|-----------|
| 4 | Résumé automatique (*Summarization*) | `sshleifer/distilbart-cnn-12-6` | ROUGE-1/2/L, BERTScore |
| 5 | Question-Réponse (*Question Answering*) | `distilbert-base-cased-distilled-squad` | Exact Match, F1, match par inclusion |
| 6 | Génération de texte (*Text Generation*) | `distilgpt2` | BLEU, perplexité, Distinct-1/2/3 |

---

## 1. Structure du projet

```
.
├── nlp_tasks.ipynb   # Notebook : code + explications + résultats + graphiques
├── README.md         # Cette documentation
└── figures/          # Graphiques PNG générés à l'exécution du notebook
```

Plan du notebook : 0. Installation → 1. Imports → 2. Données → 3. Résumé → 4. QA → 5. Génération → 6. Temps de réponse et vitesse → 7. Récapitulatif et synthèse → 8. Remarques et limites.

## 2. Installation

Prérequis : Python 3.9+ (3.10 recommandé), environ 2 Go d'espace disque pour les modèles.

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install "transformers>=4.40,<5" torch sentencepiece rouge-score sacrebleu bert-score pandas matplotlib jupyter
jupyter notebook nlp_tasks.ipynb
```

> `transformers` est limité à la version < 5 car les pipelines `summarization` et `question-answering` utilisés sont ceux de la série 4.x.

Puis exécutez les cellules dans l'ordre (**Kernel → Restart & Run All**).

**Durée indicative sur CPU : 10 à 15 minutes**, car les sections 3.4, 5.3 et 6 répètent des générations. Pour aller plus vite, réduisez `MAX_LENGTHS` (3.4), `TEMPERATURES` et `N_SEEDS` (5.3) ou `N_RUNS` (6).

### Utilisation 100 % hors ligne

Au premier lancement, les modèles (DistilBART ≈ 1,2 Go, DistilBERT ≈ 260 Mo, DistilGPT-2 ≈ 350 Mo, et `distilbert-base-uncased` ≈ 270 Mo pour BERTScore) sont téléchargés depuis le Hugging Face Hub puis mis en cache (`~/.cache/huggingface`). Ensuite, pour garantir qu'aucune requête réseau n'est faite :

```bash
# Linux / macOS
export HF_HUB_OFFLINE=1
# Windows (PowerShell)
$env:HF_HUB_OFFLINE = "1"
```

## 3. Description du code (section par section)

### 3.1 Configuration (section 1)

| Élément | Rôle |
|---------|------|
| `SEED = 42` + `set_seed` | Reproductibilité de la génération par échantillonnage |
| `DEVICE_STR` / `DEVICE_ID` | `"cuda"`/`0` si un GPU est disponible, sinon `"cpu"`/`-1` (format attendu par `pipeline`) |
| `FIG_DIR` | Dossier `figures/` où les graphiques sont sauvegardés en PNG (150 dpi) |
| `LOAD_TIMES` | Dictionnaire rempli lors du chargement de chaque modèle (secondes) |
| `sync()` | `torch.cuda.synchronize()` si GPU, sans effet sur CPU : indispensable pour chronométrer correctement un calcul GPU asynchrone |

### 3.2 Données (section 2)

Jeu de données minuscule, **intégré au notebook** (pas de téléchargement) :

| Variable | Contenu |
|----------|---------|
| `ARTICLES` | 3 courts articles (entrée du résumé et contexte du QA) |
| `REFERENCE_SUMMARIES` | Résumé de référence pour chaque article |
| `QA_SET` | Liste de tuples `(index_article, question, réponse_attendue)` |
| `GEN_PROMPTS` / `GEN_REFERENCES` | Débuts de phrases à continuer et continuation de référence (pour BLEU) |

Pour un vrai benchmark, remplacez ces listes par un extrait local de CNN/DailyMail (résumé), SQuAD (QA), etc.

### 3.3 Résumé automatique (section 3)

Modèle **abstractif** séquence-à-séquence (DistilBART).

| Fonction / élément | Description |
|--------------------|-------------|
| `summarize(texts, max_length=60, min_length=15)` | Génère un résumé par texte, avec *beam search* (`num_beams=4`), sans échantillonnage (déterministe) |
| `evaluate_rouge(predictions, references)` | Moyenne des **F1** de ROUGE-1, ROUGE-2 et ROUGE-L (`rouge-score`, avec racinisation) |
| `rouge_per_sample(predictions, references)` (3.1) | Scores ROUGE (F1) article par article, pour les graphiques |
| `rouge_full(predictions, references)` (3.2) | Précision, rappel et F1 de chaque variante ROUGE, article par article (DataFrame) → **heatmap** |
| BERTScore (3.3) | Similarité sémantique via `distilbert-base-uncased` (local). La cellule est protégée par `try/except` : si le modèle ne peut pas être téléchargé, un message s'affiche et le notebook continue (`bertscore_results = None`) |
| `MAX_LENGTHS`, `is_complete(text)` (3.4) | Balayage de `max_length` : qualité (ROUGE), **complétude** (le résumé se termine-t-il par `.`, `!` ou `?` ?) et latence |

### 3.4 Question-Réponse (section 4)

Modèle **extractif** (DistilBERT fine-tuné sur SQuAD) : il prédit l'intervalle du contexte qui contient la réponse.

| Fonction | Description |
|----------|-------------|
| `normalize_answer(s)` | Normalisation SQuAD : minuscules, suppression de la ponctuation et des articles, espaces normalisés |
| `exact_match(prediction, truth)` | 1.0 si les réponses normalisées sont identiques, sinon 0.0 |
| `f1_score(prediction, truth)` | F1 au niveau des mots communs entre prédiction et référence |
| `answer_questions(qa_set, contexts)` | Interroge le pipeline pour chaque `(contexte, question)` |
| `evaluate_qa(predictions, qa_set)` | Moyenne de EM et F1 |
| `inclusion_match(prediction, truth)` (4.2) | 1.0 si l'une des réponses (normalisées) est contenue, mot pour mot, dans l'autre. **Indulgent et non standard** : à lire avec l'EM |
| `error_type(prediction, truth)` (4.2) | Classe la réponse : `exacte`, `trop longue`, `trop courte` (erreurs de **frontière**) ou `incorrecte` |
| `qa_df` (4.2) | Tableau détaillé : prédiction, attendu, EM, F1, inclusion, **confiance** du modèle, type d'erreur, positions `start`/`end` |
| `highlight_answer(context, start, end)` (4.4) | Retourne le contexte en HTML avec le segment prédit surligné |

### 3.5 Génération de texte (section 5)

Modèle de langage **causal** (DistilGPT-2). Trois stratégies de décodage (`STRATEGIES`) :

- `greedy` : token le plus probable à chaque étape (déterministe, tend à boucler) ;
- `sampling` : échantillonnage `top_p=0.9`, `temperature=0.8` ;
- `greedy_no_repeat` : greedy avec `no_repeat_ngram_size=3` (interdit de répéter un trigramme).

| Fonction / élément | Description |
|--------------------|-------------|
| `generate_continuation(prompt, max_new_tokens=30, **gen_kwargs)` | Génère uniquement la suite du prompt (sans le répéter) |
| `perplexity(text)` | `exp(loss moyenne)` du modèle sur le texte (prompt + continuation) |
| `distinct_n(text, n)` | Ratio n-grammes uniques / n-grammes totaux (diversité) |
| `evaluate_generation(prompts, references, strategy_kwargs)` | Génère pour tous les prompts puis calcule BLEU, perplexité, Distinct-1/2/3 et nombre moyen de mots (`mean_words`) |
| `generate_ids(prompt, ...)` (5.2) | Comme `generate_continuation` mais renvoie les **ids** des tokens (prompt, nouveaux tokens) |
| `surprisal_from_ids(prompt_ids, new_ids)` (5.2) | Surprisal (−log p, en nats) de chaque nouveau token sachant tout ce qui précède |
| `trace` (5.2) | Pour chaque stratégie : texte, tokens, surprisal des tokens générés et surprisal du texte **re-tokenisé** |
| `run_sampling_config(temperature, seed)` (5.3) | Lance le sampling pour une température et une graine, renvoie BLEU, perplexité et Distinct-2 |
| `TEMPERATURES`, `N_SEEDS` (5.3) | Grille de la section 5.3 (5 températures × 5 graines par défaut) |

### 3.6 Temps de réponse et vitesse (section 6)

Mesure, pour chaque méthode (résumé, QA, et chaque stratégie de génération), avec **une requête à la fois** (batch = 1).

| Élément | Description |
|---------|-------------|
| `N_RUNS` (3) | Répétitions chronométrées par exemple |
| `WARMUP` (1) | Appels de chauffe exclus de la mesure |
| `benchmark(label, inputs, run_one, count_tokens, token_kind)` | Exécute `run_one` sur chaque exemple (chauffe puis `N_RUNS` mesures avec `time.perf_counter()`) ; renvoie les statistiques et le détail par exemple (`_per_item`) |
| `bench_df` | DataFrame des résultats, une ligne par méthode |
| `per_item_data` | Détail (tokens, latence) par exemple, utilisé pour le graphe 6.2 |

Colonnes de `bench_df` : latence moyenne, écart-type, min, max (s) ; `Exemples/s` ; `Tokens/s` (tokens **générés** pour le résumé et la génération, **analysés** pour le QA, donc non comparables entre ces deux types) ; `Type de tokens` ; `Chargement (s)`.

### 3.7 Récapitulatif et synthèse (section 7)

- `results_df` : tous les scores de qualité (tâche, modèle, métrique, score), y compris BERTScore si disponible.
- **Synthèse automatique (7.2)** : phrases calculées à partir des résultats de l'exécution (scores, résumés tronqués, erreurs de frontière du QA, meilleures stratégies, rapport de vitesse). Elles se mettent à jour si vous changez les modèles ou les paramètres.

## 4. Graphiques

Tous les graphiques sont tracés avec `matplotlib` et sauvegardés dans `figures/`.

| Fichier | Section | Type | Contenu |
|---------|---------|------|---------|
| `summarization_rouge.png` | 3.1 | Barres | ROUGE (F1) par article et longueur des textes |
| `rouge_heatmap.png` | 3.2 | **Heatmap** | Précision, rappel et F1 de ROUGE-1/2/L par article |
| `summarization_max_length.png` | 3.4 | **Courbes** | ROUGE, latence et complétude selon `max_length` |
| `qa_scores.png` | 4.1 | Barres | EM et F1 par question |
| `qa_confidence.png` | 4.3 | **Nuage de points** | Confiance du modèle vs F1, coloré par type de réponse |
| `generation_strategies.png` | 5.1 | Barres | BLEU, perplexité, Distinct-1/2/3, mots générés par stratégie |
| `generation_surprisal.png` | 5.2 | **Courbes** | Surprisal de chaque token généré, par stratégie |
| `generation_temperature.png` | 5.3 | **Boîtes à moustaches + courbe** | Variabilité selon la graine, effet de la température, compromis fluidité/diversité |
| `performance_timing.png` | 6.1 | Barres | Latence, exemples/s, tokens générés/s |
| `performance_latency_vs_tokens.png` | 6.2 | **Nuage de points + droites** | Latence en fonction du nombre de tokens |
| `summary_metrics.png` | 7.1 | Barres horizontales | Métriques entre 0 et 1 sur un même axe |

La section 4.4 affiche en plus les **réponses surlignées dans le contexte** (HTML, non sauvegardé en PNG).

## 5. Métriques d'évaluation

| Métrique | Tâche | Intervalle | Meilleur si | Ce qu'elle mesure |
|----------|-------|------------|-------------|-------------------|
| ROUGE-1 / ROUGE-2 | Résumé | 0 – 1 | plus haut | Recouvrement de mots / de paires de mots avec la référence |
| ROUGE-L | Résumé | 0 – 1 | plus haut | Plus longue sous-séquence commune |
| BERTScore (P, R, F1) | Résumé | ≈ 0 – 1 (valeurs brutes élevées) | plus haut | Similarité sémantique via embeddings ; à utiliser pour **comparer** des systèmes |
| Exact Match | QA | 0 – 1 | plus haut | Réponse strictement identique à la référence |
| F1 | QA | 0 – 1 | plus haut | Recouvrement de mots (accepte les réponses partielles) |
| Match par inclusion | QA | 0 – 1 | plus haut | Indulgent : l'une des réponses contient l'autre (non standard) |
| BLEU | Génération | 0 – 100 | plus haut | Recouvrement de n-grammes avec une continuation de référence |
| Perplexité | Génération | ≥ 1 | plus bas | Probabilité du texte selon le modèle |
| Distinct-1 / 2 / 3 | Génération | 0 – 1 | plus haut | Diversité (proportion de n-grammes uniques) |
| Surprisal | Génération | ≥ 0 nats | plus bas | « Surprise » du modèle à chaque token |

Métriques de performance :

| Métrique | Unité | Meilleur si | Ce qu'elle mesure |
|----------|-------|-------------|-------------------|
| Latence (temps de réponse) | secondes / appel | plus bas | Durée pour traiter un exemple |
| Exemples/s | exemples / s | plus haut | Capacité de traitement (batch = 1) |
| Tokens/s | tokens / s | plus haut | Vitesse de génération (ou d'analyse pour le QA) |
| Chargement | secondes | plus bas | Temps d'initialisation du modèle (inclut le téléchargement au premier lancement) |

> Les temps dépendent du matériel (CPU vs GPU) et de la longueur des textes : comparez les méthodes **entre elles sur la même machine**.

## 6. Personnalisation

```python
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"          # ex. "facebook/bart-large-cnn" (plus précis, plus lourd)
QA_MODEL  = "distilbert-base-cased-distilled-squad"  # ex. "deepset/roberta-base-squad2"
GEN_MODEL = "distilgpt2"                             # ex. "gpt2-medium"
```

Ajouter une stratégie de génération : compléter le dictionnaire `STRATEGIES`, par exemple `"beam": dict(do_sample=False, num_beams=5, no_repeat_ngram_size=2)`. Elle apparaît automatiquement dans les évaluations, le chronométrage et le récapitulatif.

## 7. Limites

- Le jeu de données (3 articles, 6 questions, 3 prompts) **illustre la méthode** ; les scores ne sont pas un benchmark. Évaluez sur au moins quelques centaines d'exemples pour des conclusions fiables.
- ROUGE et BLEU mesurent un **recouvrement lexical**, pas le sens ; BERTScore y remédie en partie, mais ses valeurs brutes sont naturellement élevées.
- Le match par inclusion est indulgent : il peut accepter à tort une réponse trop courte. Il sert à repérer les erreurs de frontière, pas à remplacer l'EM.
- En génération, une seule métrique ne suffit pas : la perplexité seule ne détecte pas les répétitions, Distinct-n ignore les retours à la ligne, et la perplexité du texte décodé peut différer de celle des tokens réellement générés (section 5.2). Le sampling varie avec la graine (section 5.3).
- Les modèles choisis sont **petits** pour tourner sur CPU ; leur qualité est inférieure à celle de modèles plus grands.
- Les modèles sont en **anglais** ; pour du français, utilisez par exemple un modèle multilingue ou fine-tuné sur du français (`mT5`, `camembert`, `barthez`).

## 8. Dépannage

| Problème | Solution |
|----------|----------|
| Message `ERROR: pip's dependency resolver ... gradio / diffusers` dans Colab | Sans conséquence : ces paquets préinstallés ne sont pas utilisés par le notebook. Si une cellule échoue ensuite avec un `ImportError`, redémarrez la session et relancez à partir de la cellule 1 |
| `KeyError: 'summarization'` ou tâche inconnue | Vérifier `transformers<5` : `pip install "transformers>=4.40,<5"` |
| `BERTScore indisponible` | Le modèle `distilbert-base-uncased` n'a pas pu être téléchargé ou `bert-score` n'est pas installé ; le reste du notebook fonctionne sans |
| Téléchargement impossible (réseau) | Télécharger les modèles une fois sur une machine connectée et copier `~/.cache/huggingface` |
| Exécution trop longue sur CPU | Réduire `MAX_LENGTHS`, `TEMPERATURES`, `N_SEEDS`, `N_RUNS`, ou activer un GPU (Colab : Exécution → Modifier le type d'exécution) |
| Mémoire insuffisante | Réduire `max_length` / `max_new_tokens`, ou rester sur les modèles *distil* |
| Résultats différents d'une exécution à l'autre | Normal pour `sampling` ; fixer `SEED` et relancer la cellule |