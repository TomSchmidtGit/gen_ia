import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Configuration de l'application Streamlit
st.set_page_config(page_title="Projet IA - Génération d'Images", layout="wide")
st.title("🖼️ Génération d'Images avec Stable Diffusion v1.5")

# 🎨 Amélioration de l'interface utilisateur
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    .stTextInput > div > input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Chargement du modèle (cela peut prendre un peu de temps la première fois)
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # Nom correct du modèle
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            variant="fp16",
            use_auth_token=True  # Permet l'utilisation de ton Token Hugging Face
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe.to(device)
        return pipe
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

pipe = load_model()

# 📌 Message d'avertissement si on utilise le CPU
if torch.backends.mps.is_available():
    st.success("✅ GPU MPS détecté ! Le modèle est chargé sur GPU.")
else:
    st.warning("⚠️ Aucun GPU détecté. Le modèle est chargé sur le CPU, ce qui peut être lent.")

# 🎯 Entrée utilisateur
st.subheader("📝 Prompt de génération")
prompt = st.text_input("💬 Entrez votre prompt :", value="A beautiful sunset over a mountain landscape, photorealistic")

# 📌 Paramètres avancés
st.sidebar.header("🔧 Paramètres avancés")
guidance_scale = st.sidebar.slider("🎛 Proximité avec le prompt (Plus la valeur est haute plus la génération sera proche du prompt)", 1.0, 20.0, 7.5)
num_inference_steps = st.sidebar.slider("📏 Nomre d'étapes pour affiner l'image (plus d'étapes = plus longué génération mais plus proche du prompt)", 10, 100, 50)

# 🎨 Bouton pour générer l'image
if st.button("🎨 Générer l'image"):
    if not prompt.strip():
        st.warning("⚠️ Veuillez entrer un prompt valide.")
    else:
        with st.spinner("🖌️ Génération en cours..."):
            try:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                pipe.to(device)
                
                image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

                # 📌 Affichage de l'image générée
                st.image(image, caption="✅ Image générée avec succès", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération : {e}")