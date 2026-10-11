# RAGAS : notes et faisabilité pour ce projet

Document de référence sur RAGAS (le framework d'évaluation de RAG) et analyse
honnête de son applicabilité à notre pipeline. Rédigé en complément des
évaluations RAG déjà en place (`rag_pearson_evaluation.py` pour le retrieval,
`rag_utility_ab_synthesis.md` pour l'utilité, et la garantie de citation
déterministe du Sep Agent).

---

## 1. Qu'est-ce que RAGAS

RAGAS (Retrieval-Augmented Generation Assessment) est un framework open source
dédié à l'évaluation des pipelines RAG. Son idée centrale : un système RAG a
trois éléments et chaque métrique isole la qualité d'un maillon.

Le triplet RAG :
- **question** (la requête),
- **context** (les passages récupérés par le retriever),
- **answer** (la réponse générée par le LLM),
- et optionnellement **ground_truth** (la réponse ou les passages de référence).

RAGAS mesure les relations entre ces éléments. Beaucoup de ses métriques sont
dites **reference-free** (sans réponse de référence) : elles utilisent un LLM
comme juge pour estimer la cohérence entre question, contexte et réponse.

---

## 2. Les métriques principales

### Côté récupération (retriever)
- **Context Precision** : parmi les passages récupérés, quelle proportion est
  réellement pertinente, et les pertinents sont-ils bien classés en tête.
- **Context Recall** : les passages récupérés couvrent-ils toute l'information
  nécessaire pour répondre. Nécessite une **réponse de référence** (ground_truth).
- **Context Entities Recall** : proportion des entités de la référence
  effectivement présentes dans le contexte récupéré.

### Côté génération (LLM)
- **Faithfulness (fidélité)** : la réponse est-elle entièrement soutenue par le
  contexte récupéré. Le juge décompose la réponse en affirmations atomiques et
  vérifie chacune contre le contexte. C'est la métrique anti-hallucination.
- **Answer Relevancy** : la réponse répond-elle bien à la question (pénalise le
  hors-sujet, l'incomplétude, la redondance).
- **Answer Correctness** : exactitude de la réponse vs la référence (combine
  similarité sémantique et exactitude factuelle). Nécessite un ground_truth.
- **Answer Semantic Similarity** : similarité d'embedding entre réponse et
  référence.

Résumé : **precision/recall** jugent le retriever, **faithfulness/relevancy**
jugent la génération, **correctness/similarity** exigent une réponse de référence.

---

## 3. Comment ça marche techniquement

- **Un juge LLM est obligatoire** pour faithfulness, answer relevancy, context
  precision. Par défaut RAGAS utilise l'API OpenAI, mais le juge est
  configurable (tout LLM compatible LangChain, y compris un modèle local via
  Ollama, ou un backend cloud).
- **Un modèle d'embedding** est requis pour les métriques de similarité.
- **Dépendances** : `ragas` s'appuie sur **LangChain** et le format `datasets`
  (HuggingFace). Les données d'entrée sont un tableau de lignes
  `{question, contexts, answer, ground_truth}`.
- Sortie : un score moyen par métrique (0 à 1) sur le jeu d'évaluation.

---

## 4. Forces et limites générales

Forces :
- Métriques **standard et reconnues** dans la littérature RAG (crédibilité dans
  un rapport).
- Sépare proprement qualité du **retrieval** et qualité de la **génération**.
- Plusieurs métriques **sans réponse de référence**, utile quand on n'a pas de
  gold answers.

Limites :
- **Bruit du juge LLM** : les scores dépendent du modèle juge et de sa
  température ; instables sur de petits jeux.
- **Coût / dépendance** : nécessite un LLM juge (API payante ou modèle local
  fort) et ajoute LangChain + datasets.
- **Biais** : un juge faible sur du texte clinique spécialisé peut mal évaluer.
- **Non déterministe** : deux exécutions peuvent donner des scores différents.

---

## 5. Faisabilité pour NOTRE projet

### 5.1 Forme de notre RAG (rappel)
Notre RAG est une **couche du Sep Agent** :
- la **question** n'est pas une phrase d'utilisateur mais une requête construite
  automatiquement à partir des **flags** du cas (via `_FLAG_QUERY_PHRASES`) ;
- le **context** = top-k chunks d'une petite KB curée (**83 chunks / 3 guides**,
  embedding `all-MiniLM-L6-v2`) ;
- la **answer** = l'impression clinique **multimodale** du Sep Agent, qui cite
  AUSSI les findings des agents ECG/MRI/patient, pas seulement les chunks RAG ;
- pas de **ground_truth** (pas de réponse de référence écrite).

### 5.2 Adéquation métrique par métrique

| Métrique RAGAS | Adaptée ici ? | Pourquoi |
|---|---|---|
| Context Precision | Oui | On a déjà 25 requêtes étiquetées (dont 14 exploitables sur l'index actuel) → base de gold retrieval. |
| Context Recall | Partiel | Exige un ground_truth ; à écrire à la main. |
| Faithfulness | Oui mais à cadrer | Notre réponse cite aussi les findings modalités : « fidélité au contexte RAG seul » pénaliserait à tort ces citations légitimes. Il faut définir le contexte = chunks RAG + findings, ou évaluer une sous-réponse RAG-only. |
| Answer Relevancy | Bancale | Notre requête est une liste de flags, pas une question ; il faudrait reformuler en questions. |
| Answer Correctness / Similarity | Non (pour l'instant) | Aucune réponse de référence disponible. |

### 5.3 Ce qu'on a DÉJÀ pour le RAG
- **Retrieval** : `rag_pearson_evaluation.py` (Pearson r, top-1, séparation) sur
  25 requêtes étiquetées. Général r 0.770 / top-1 13/14 ; clinique r 0.770 /
  14/14.
- **Utilité** : A/B (RAG off vs on) dans `rag_utility_ab_synthesis.md` (le RAG
  n'aide qu'avec une KB propre et bien extraite).
- **Fidélité, version DURE** : `parse_sep_agent_response` **rejette
  déterministiquement** toute citation "knowledge" absente des chunks récupérés
  (retry forcé). C'est une **garantie**, plus forte que le score *souple* jugé
  par LLM de RAGAS.

Point clé : nous **n'avons aujourd'hui aucun juge LLM** ; toutes nos évaluations
sont déterministes, métriques ou par embedding. **RAGAS introduirait le premier
juge LLM du projet.**

### 5.4 Coûts spécifiques à assumer
1. **Juge LLM** : MedGemma 4B est trop faible pour juger de façon fiable (plafond
   « G2 » déjà documenté) → il faudrait utiliser le **backend Groq** déjà intégré
   comme juge (données = guidelines publiques, donc sortir de la machine est
   acceptable) ou un modèle frontière.
2. **Dépendances** : `ragas` + `langchain` + `datasets`, alors que le projet a
   **délibérément évité LangChain**. Arbitrage réel.
3. **Bruit** : petit jeu (14 requêtes, 83 chunks) → scores RAGAS instables ;
   à interpréter avec prudence.

### 5.5 Recommandation
- **Si l'objectif est d'afficher le label « RAGAS » dans le rapport** :
  intégration ciblée sur **Context Precision + Faithfulness**, juge = **Groq**,
  en réutilisant les 14-25 requêtes existantes, présentée en **complément** (pas
  en remplacement) des évals actuelles. Ne pas la survendre vu le bruit.
- **Si l'objectif est la rigueur/reproductibilité** : rester sur l'existant
  (Pearson retrieval + A/B utilité + garantie de citation déterministe) et
  documenter que l'absence de juge LLM est un **choix** (reproductible,
  auditable), pas un manque. Éventuellement ajouter un **script maison de
  faithfulness** (décomposer l'impression en affirmations, vérifier que chacune
  cite un chunk réel) sans introduire RAGAS/LangChain.

### 5.6 Verdict en une phrase
RAGAS est **faisable et apporterait des métriques RAG reconnues**, mais il
n'ajoute pas de sécurité nouvelle (on garantit déjà la fidélité de citation de
façon déterministe) ; il apporte surtout un **chiffre présentable** au prix d'un
premier juge LLM, de nouvelles dépendances, et d'un bruit de mesure sur un petit
jeu.

---

## 6. Esquisse d'intégration minimale (si retenue)

1. Construire un petit jeu `{question, contexts, answer}` :
   - `question` = les phrases de requête générées depuis les flags d'un cas,
   - `contexts` = les chunks renvoyés par `rag_knowledge_base.retrieve`,
   - `answer` = `SepAgentDecision.summary` (avec `use_rag=True`).
2. Configurer le juge RAGAS sur le backend Groq (clé `GROQ_API_KEY`), l'embedding
   sur un modèle local.
3. Lancer d'abord **Context Precision** (n'exige pas de référence) et
   **Faithfulness** (en cadrant le contexte, cf. 5.2).
4. Comparer aux chiffres Pearson existants et documenter le bruit sur plusieurs
   exécutions.
