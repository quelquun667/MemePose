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

Tous les paramètres ajustables sont en haut de [main.py](main.py) :

| Constante | Valeur par défaut | Description |
|---|---|---|
| `MIN_DETECTION_CONFIDENCE` | `0.7` | Confiance minimale pour détecter une main |
| `MIN_TRACKING_CONFIDENCE` | `0.6` | Confiance minimale pour continuer le tracking |
| `DISPLAY_HOLD_FRAMES` | `15` | Frames où le mème reste affiché après la fin du geste |
| `MEME_FILENAME` | `hamster.png` | Fichier PNG à incruster |

---

## Licence

MIT — voir [LICENSE](LICENSE) pour les détails.
