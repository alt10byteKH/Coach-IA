import json
import os
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

MAX_SECONDS = 20
MODEL = "gemini-2.5-flash"
PL = mp.solutions.pose.PoseLandmark

# (épaule, coude, poignet, hanche, genou, cheville) par côté
SIDES = {
    "droite": [PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST,
               PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE],
    "gauche": [PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST,
               PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE],
}

PROMPT = """Tu es un entraîneur de tennis diplômé (DE/BE) et un expert en biomécanique.
Tu reçois {n} images chronologiques d'un {coup} ({moments}), ainsi que des mesures
calculées automatiquement par détection de pose (fiables à ±10° environ).

CONTEXTE
- Niveau déclaré par le joueur : {niveau}
- Main de frappe détectée : {main}
- Mesures à l'impact : {metrics}

RÈGLES STRICTES
- Base-toi uniquement sur ce qui est visible. Si un élément n'est pas visible
  (angle de caméra, flou, joueur coupé), écris "non observable" : n'invente rien.
- Chaque défaut doit citer une preuve visuelle ou chiffrée.
- Le classement est une estimation FFT indicative : donne une fourchette
  (ex. "30/2 à 30/1") et un niveau de confiance, jamais une valeur unique.
- Ton : encourageant, concret, vocabulaire simple. Réponds en français.

Réponds UNIQUEMENT en JSON valide avec cette structure :
{{
  "qualite_video": {{"exploitable": true, "remarques": "..."}},
  "classement": {{"fourchette": "...", "confiance": "faible|moyenne|haute", "justification": "..."}},
  "points_forts": ["..."],
  "defauts": [{{"probleme": "...", "preuve": "...", "gravite": "mineure|moyenne|majeure", "correction": "..."}}],
  "priorite_numero_1": "le seul point à travailler en premier",
  "exercices": [{{"nom": "...", "description": "...", "duree": "...", "objectif": "..."}}],
  "plan_2_semaines": [{{"semaine": 1, "focus": "...", "seances": ["...", "..."]}},
                      {{"semaine": 2, "focus": "...", "seances": ["...", "..."]}}]
}}"""


def angle(a, b, c):
    v1, v2 = np.array(a) - np.array(b), np.array(c) - np.array(b)
    cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_poses(path):
    """Passe 1 : positions des articulations pour chaque image (léger en mémoire)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total / fps > MAX_SECONDS:
        cap.release()
        raise ValueError(f"Vidéo trop longue (max {MAX_SECONDS} s).")
    poses = []
    with mp.solutions.pose.Pose(model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (480, int(480 * h / w)))
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                poses.append(np.array([[p.x, p.y] for p in res.pose_landmarks.landmark]))
            else:
                poses.append(None)
    cap.release()
    return fps, poses


def find_impact(poses, fps):
    """Impact = pic de vitesse du poignet. Renvoie (index, main de frappe)."""
    best = (0.0, None, None)
    for side, ids in SIDES.items():
        w = ids[2].value
        speeds = [0.0] * len(poses)
        for i in range(1, len(poses)):
            if poses[i] is not None and poses[i - 1] is not None:
                speeds[i] = np.linalg.norm(poses[i][w] - poses[i - 1][w]) * fps
        speeds = np.convolve(speeds, np.ones(3) / 3, mode="same")
        i = int(np.argmax(speeds))
        if speeds[i] > best[0]:
            best = (speeds[i], i, side)
    if best[1] is None:
        raise ValueError("Aucun joueur détecté. Filme le joueur en entier, bien visible.")
    return best[1], best[2]


def impact_metrics(pose, side):
    sh, el, wr, hip, kn, an = [pose[i.value] for i in SIDES[side]]
    other = "gauche" if side == "droite" else "droite"
    o_hip, o_kn, o_an = [pose[SIDES[other][i].value] for i in (3, 4, 5)]
    mid_sh = (pose[PL.LEFT_SHOULDER.value] + pose[PL.RIGHT_SHOULDER.value]) / 2
    mid_hip = (pose[PL.LEFT_HIP.value] + pose[PL.RIGHT_HIP.value]) / 2
    trunk = mid_sh - mid_hip
    lean = float(np.degrees(np.arctan2(abs(trunk[0]), abs(trunk[1]) + 1e-9)))
    return {
        "angle_coude_frappe": round(angle(sh, el, wr)),
        "angle_genou_cote_frappe": round(angle(hip, kn, an)),
        "angle_genou_oppose": round(angle(o_hip, o_kn, o_an)),
        "inclinaison_tronc_vs_vertical": round(lean),
    }


def grab(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def key_frames(path, impact, fps, n):
    moments = [("préparation", -0.5), ("armé", -0.2), ("impact", 0), ("fin de geste", 0.4)]
    imgs, names = [], []
    for name, dt in moments:
        idx = min(max(int(impact + dt * fps), 0), n - 1)
        img = grab(path, idx)
        if img:
            imgs.append(img)
            names.append(name)
    return imgs, names


def ask_gemini(api_key, imgs, names, coup, niveau, main, metrics):
    client = genai.Client(api_key=api_key)
    prompt = PROMPT.format(
        n=len(imgs), coup=coup, moments=", ".join(names),
        niveau=niveau, main=main, metrics=json.dumps(metrics),
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt, *imgs],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.3
        ),
    )
    return json.loads(resp.text)


def show_report(r):
    q = r.get("qualite_video", {})
    if not q.get("exploitable", True):
        st.warning(f"Vidéo difficile à analyser : {q.get('remarques', '')}")
    c = r.get("classement", {})
    st.metric("Classement estimé (indicatif)", c.get("fourchette", "?"),
              f"confiance {c.get('confiance', '?')}", delta_color="off")
    st.caption(c.get("justification", ""))
    st.success(f"🎯 Priorité n°1 : {r.get('priorite_numero_1', '')}")
    st.subheader("Points forts")
    for p in r.get("points_forts", []):
        st.markdown(f"- ✅ {p}")
    st.subheader("Points à corriger")
    icon = {"mineure": "🟡", "moyenne": "🟠", "majeure": "🔴"}
    for d in r.get("defauts", []):
        with st.expander(f"{icon.get(d.get('gravite'), '⚪')} {d.get('probleme', '')}"):
            st.markdown(f"**Preuve :** {d.get('preuve', '')}")
            st.markdown(f"**Correction :** {d.get('correction', '')}")
    st.subheader("Exercices")
    for e in r.get("exercices", []):
        st.markdown(f"**{e.get('nom')}** ({e.get('duree')}) — {e.get('description')}")
    st.subheader("Plan sur 2 semaines")
    for s in r.get("plan_2_semaines", []):
        st.markdown(f"**Semaine {s.get('semaine')} — {s.get('focus')}**")
        for seance in s.get("seances", []):
            st.markdown(f"- {seance}")


# ---------------- Interface ----------------
st.set_page_config(page_title="AI Tennis Coach", page_icon="🎾")
st.title("🎾 AI Tennis Coach")
st.caption("Détection de pose + coach IA. Estimation indicative, pas un avis officiel.")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.text_input("Clé API Google AI Studio (gratuite)", type="password")

col1, col2 = st.columns(2)
coup = col1.selectbox("Coup filmé", ["coup droit", "revers", "service"])
niveau = col2.selectbox("Ton niveau", ["débutant", "intermédiaire", "avancé", "je ne sais pas"])

up = st.file_uploader(f"Vidéo (max {MAX_SECONDS} s, joueur en entier)", type=["mp4", "mov"])

if up and st.button("Analyser", type="primary"):
    if not api_key:
        st.error("Clé API manquante.")
        st.stop()
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
            f.write(up.read())
            path = f.name
        with st.spinner("Détection des mouvements..."):
            fps, poses = extract_poses(path)
            impact, main = find_impact(poses, fps)
            if poses[impact] is None:
                raise ValueError("Pose introuvable à l'impact, filme plus près ou mieux éclairé.")
            metrics = impact_metrics(poses[impact], main)
            imgs, names = key_frames(path, impact, fps, len(poses))
        st.image(imgs, caption=names, width=150)
        with st.spinner("Le coach analyse..."):
            report = ask_gemini(api_key, imgs, names, coup, niveau, main, metrics)
        show_report(report)
    except json.JSONDecodeError:
        st.error("Réponse IA illisible, relance l'analyse.")
    except Exception as e:
        st.error(f"Erreur : {e}")
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
