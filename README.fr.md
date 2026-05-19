<div align="right">
  <a href="README.md">🇬🇧 Read in English</a>
</div>

<div align="center">
  <h1>✌️ MemePose</h1>
  <p><strong>Incrustation de mèmes en temps réel déclenchée par un geste de la main</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-green?logo=google&logoColor=white" alt="MediaPipe">
    <img src="https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/licence-MIT-yellow" alt="Licence">
  </p>
</div>

---

## C'est quoi MemePose ?

MemePose détecte les gestes de la main en temps réel via la webcam et affiche une image mème à l'écran dès qu'un geste spécifique est reconnu.

Fais le **signe Peace ✌️** → un mème apparaît avec la mention **"HAMSTER DETECTED!"**

Le squelette de la main (21 points de repère) est affiché en permanence pour visualiser ce que l'IA suit.

---

## Fonctionnalités

- **Tracking de la main en temps réel** via MediaPipe Hand Landmarker (API Tasks)
- **Mode miroir** — les mouvements sont naturels à l'écran
- **Incrustation PNG avec transparence** — les mèmes s'incrustent sans fond parasite
- **Anti-clignotement** — le mème reste visible quelques frames après la fin du geste
- **Assets plug-and-play** — dépose n'importe quel PNG dans `assets/` et renomme-le

---

## Prérequis

- Python 3.10+
- Une webcam

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/your-username/MemePose.git
cd MemePose

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt
```

> Le modèle de détection MediaPipe (~3 Mo) est téléchargé automatiquement au premier lancement.

---

## Utilisation

```bash
python main.py
```

| Touche | Action |
|--------|--------|
| `Q` ou `Échap` | Quitter |

---

## Structure du projet

```
MemePose/
├── main.py                   # Point d'entrée de l'application
├── generate_placeholder.py   # Utilitaire : génère un mème PNG placeholder
├── requirements.txt
├── .gitignore
└── assets/
    └── hamster.png           # Mème PNG (BGRA avec transparence)
```

---

## Ajouter ses propres mèmes

1. Prépare un **fichier PNG avec fond transparent** (BGRA / RGBA).
2. Dépose-le dans `assets/`.
3. Modifie `MEME_FILENAME` dans [main.py](main.py) :

```python
MEME_FILENAME = "ton_meme.png"
```

Pas encore de mème ? Lance le générateur de placeholder :

```bash
python generate_placeholder.py
```

---

## Comment fonctionne la détection de geste ?

MediaPipe retourne 21 points normalisés (x, y, z) pour chaque main détectée.

Le **signe Peace ✌️** est validé quand :
- Le bout de l'index (point 8) est **au-dessus** de sa deuxième articulation (point 6)
- Le bout du majeur (point 12) est **au-dessus** de sa deuxième articulation (point 10)
- Le bout de l'annulaire (point 16) est **en dessous** de sa deuxième articulation (point 14)
- Le bout de l'auriculaire (point 20) est **en dessous** de sa deuxième articulation (point 18)

*"Au-dessus" = valeur Y plus petite en coordonnées image (l'origine est en haut à gauche).*

---

## Configuration

Tous les paramètres ajustables sont en haut de [main.py](main.py). Si les détections semblent imprécises, commence par les seuils EAR des expressions.

### Général

| Constante | Valeur par défaut | Description |
|---|---|---|
| `CONFIRM_FRAMES` | `5` | Nombre de frames consécutives où la pose doit être tenue avant déclenchement — augmenter si trop sensible, réduire si lent |
| `DISPLAY_HOLD_FRAMES` | `15` | Frames où le mème reste affiché après la fin du geste |
| `MIN_HAND_CONFIDENCE` | `0.7` | Confiance minimale pour détecter une main |
| `MIN_TRACK_CONFIDENCE` | `0.6` | Confiance minimale pour continuer le tracking |
| `MIN_FACE_CONFIDENCE` | `0.5` | Confiance minimale pour détecter un visage |

### Expressions faciales — EAR (Eye Aspect Ratio)

L'EAR mesure l'ouverture d'un œil : `hauteur verticale / largeur horizontale`. Un œil fermé vaut environ `0.0` ; un œil normalement ouvert se situe entre `0.25` et `0.35`. **Ces valeurs varient beaucoup d'une personne à l'autre** — quelqu'un avec de grands yeux naturels aura un EAR au repos plus élevé que la moyenne.

> **Comment calibrer :** ajoute un `print(ear_l, ear_r)` dans `is_squinting()` et observe la console en faisant chaque expression. Utilise ces valeurs pour régler tes seuils.

| Constante | Valeur par défaut | Ce qu'elle contrôle |
|---|---|---|
| `SQUINT_EAR_MIN` | `0.12` | Borne basse du plissement — en dessous, l'œil est considéré fermé (zone clignement) |
| `SQUINT_EAR_MAX` | `0.22` | Borne haute du plissement — au-dessus, l'œil est considéré normalement ouvert |
| `WINK_EAR_CLOSED` | `0.10` | Un œil doit être en dessous de ce seuil pour valider un clin d'œil |
| `WINK_EAR_OPEN` | `0.20` | L'autre œil doit dépasser ce seuil pour confirmer qu'il est ouvert |
| `WIDE_EAR_MIN` | `0.48` | Les deux yeux doivent dépasser ce seuil pour déclencher « gros yeux » — **à augmenter si ça se déclenche au repos** |

**Exemple — grands yeux naturels :** si ton EAR au repos tourne autour de `0.38`, passe `WIDE_EAR_MIN` à `0.52` pour que seul un air vraiment surpris déclenche l'expression.

**Exemple — yeux petits / en amande :** si le plissement ne se déclenche jamais, baisse `SQUINT_EAR_MAX` de `0.22` à `0.18`.

### Gestes de mouvement

| Constante | Valeur par défaut | Description |
|---|---|---|
| `WAVE_MIN_DELTA` | `0.10` | Déplacement vertical minimal de chaque main pour valider une vague |
| `SCUBA_MIN_X_RANGE` | `0.20` | Déplacement horizontal minimal de la main qui ondule pour le scuba |
| `SCUBA_PINCH_MAX_DIST` | `0.09` | Distance 3D maximale pouce–index pour valider un pincement |

---

## Licence

MIT — voir [LICENSE](LICENSE) pour les détails.
