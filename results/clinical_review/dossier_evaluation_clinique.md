# Dossier d'évaluation clinique - demande d'avis spécialisé

**Outil d'aide à l'interprétation cardiovasculaire multimodale**
*À l'attention de Dr Leila Abid - demande d'avis spécialisé*
*Projet de recherche académique - prototype*

---

## Présentation du projet

Ce document s'inscrit dans un **stage d'été réalisé au CRNS (Centre de Recherche en Numérique de
Sfax)** portant sur un **système d'aide à l'interprétation en cardiologie**. L'idée : combiner plusieurs examens d'un même
patient - **ECG, IRM cardiaque et données cliniques** - à l'aide de méthodes d'intelligence
artificielle, pour **mettre en évidence, de façon traçable, les éléments qui méritent une revue
clinique**. L'outil n'a pas vocation à remplacer le médecin, mais à l'assister.

Le prototype a été développé et évalué sur des **bases de données publiques anonymisées** ; il
fonctionne aujourd'hui de bout en bout (de l'examen brut jusqu'à une synthèse intégrée). Avant d'aller
plus loin, il lui manque l'étape essentielle d'une **relecture par une cardiologue** - c'est tout
l'objet de cette sollicitation.

---

## 1. Ce que nous attendons précisément de vous

L'outil repose sur une série de **signalements automatiques** (« flags ») : à chaque fois qu'une
valeur mesurée franchit un seuil, l'outil lève une **alerte « à faire revoir par un clinicien »**.

Ces signalements sont le cœur de l'outil, et c'est **exactement là que votre expertise nous est
indispensable**. Pour chacun, nous aimerions savoir :

1. **Le paramètre est-il cliniquement pertinent** pour justifier une alerte ?
2. **Le seuil retenu est-il juste** (ni trop sensible, ni trop laxiste) ? Sinon, quelle valeur ?
3. **Déclencher une alerte « à revoir » dans ce cas est-il approprié** et sûr ?

Une **grille de validation flag par flag** figure en section 3 : c'est le document principal à
renseigner. La lecture prend ~10 minutes.

---

## 2. Comprendre un « signalement » avant de le juger

Un **signalement** est une règle simple et reproductible : *si* une valeur mesurée franchit un seuil
fixe, *alors* l'outil l'ajoute à une liste « à revoir ». Il ne s'agit **pas** d'un diagnostic.

> **Principe directeur :** l'outil **ne pose pas de diagnostic autonome**. Il mesure, il signale ce
> qui mérite une revue humaine, et relie chaque signalement à la donnée qui l'a déclenché. Il est
> conçu comme un **support à l'interprétation**, jamais comme un substitut au jugement médical.

Deux conséquences importantes pour votre lecture :

- Comme ce sont des **règles déterministes** (et non une « boîte noire »), un signalement est toujours
  **reproductible et explicable** - mais il n'est cliniquement utile que si **le seuil est bien
  choisi**. D'où votre relecture.
- Certains signalements sont **volontairement sensibles** (mieux vaut sur-alerter que rater) : ils
  peuvent donc générer des **fausses alertes**. Nous aimerions votre avis sur ce compromis.

---

## 3. Grille de validation des signalements  *(partie principale)*

Pour chaque signalement : notez la **pertinence de 1 à 5**, indiquez si le **seuil** vous convient
(« oui » ou la valeur que vous proposez), et ajoutez un commentaire si besoin.

### 3.1 Signalements ECG
| Signalement (déclenché quand) | Pertinent ? (1-5) | Seuil approprié ? (oui / valeur proposée) | Commentaire |
|---|:--:|---|---|
| Tachycardie (FC > 120 bpm) | | | |
| Bradycardie (FC < 60 bpm) | | | |
| Rythme irrégulier (variabilité R-R élevée) | | | |
| QRS large (durée ≥ 120 ms) | | | |
| Hypertrophie ventriculaire gauche - Sokolow-Lyon (≥ 3,5 mV) | | | |
| Séquelle d'infarctus inférieur (ondes Q pathologiques ≥ 2 dérivations inférieures) | | | |
| Orientation évoquée par classe (probabilité > 50 % : conduction, hypertrophie, infarctus, repolarisation) | | | |

### 3.2 Signalements IRM cardiaque
| Signalement (déclenché quand) | Pertinent ? (1-5) | Seuil approprié ? (oui / valeur proposée) | Commentaire |
|---|:--:|---|---|
| Fraction d'éjection VG abaissée (FEVG < 45 %) | | | |
| Fraction d'éjection VG élevée (FEVG > 65 %) | | | |
| Fraction d'éjection VD abaissée (FEVD < 45 %) | | | |
| Orientation morphologique (probabilité > 50 % : dilatée, hypertrophique, séquelle d'infarctus, anomalie VD) | | | |

### 3.3 Signalements - données cliniques du patient
| Signalement (déclenché quand) | Pertinent ? (1-5) | Seuil approprié ? (oui / valeur proposée) | Commentaire |
|---|:--:|---|---|
| Scintigraphie thallium anormale (défect réversible / fixe) | | | |
| Atteinte pluritronculaire (≥ 1 vaisseau atteint) | | | |
| Angor d'effort (angor induit par l'effort) | | | |
| Sous-décalage ST à l'effort (oldpeak ≥ 0,8 mm) | | | |
| FC maximale réduite (< 91 % de 208 − 0,7 × âge) | | | |
| Hypertension de repos (PA ≥ 140 mmHg) | | | |
| Hypercholestérolémie (≥ 240 mg/dL) | | | |

---

## 4. Comment nous transmettre votre avis

Renseignez les **grilles de la section 3** (pertinence, seuil, commentaire) directement dans ce
document, ou par tout retour écrit/oral qui vous conviendra. Un **commentaire libre** - sur un
signalement manquant, un seuil à revoir, ou une réserve majeure - est également le bienvenu.

**Merci sincèrement pour le temps et l'expertise que vous accorderez à cette relecture.**
