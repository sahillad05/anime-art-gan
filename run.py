import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

st.set_page_config(page_title="GAN Art Generator", layout="centered")

st.title("🎨 GAN Art Generator")
st.write("Generate art using DCGAN or conditional anime faces using CGAN.")

# ===============================
# Session state to keep images
# ===============================
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "generated_caption" not in st.session_state:
    st.session_state.generated_caption = ""

# ===============================
# 1️⃣ Choose model
# ===============================
model_choice = st.selectbox("Choose Model:", ["DCGAN", "CGAN"])

# ===============================
# 2️⃣ Load model with spinner
# ===============================
@st.cache_resource(show_spinner=True)
def load_generator(path):
    return tf.keras.models.load_model(path, compile=False)

# ===============================
# 3️⃣ DCGAN Section
# ===============================
if model_choice == "DCGAN":
    st.subheader("DCGAN Generator")
    dcgan_path = st.text_input("DCGAN generator path", value="generator.h5")
    with st.spinner("Loading DCGAN generator..."):
        generator = load_generator(dcgan_path)

    num_images = st.slider("Number of images", 1, 6, 1)
    seed = st.number_input("Random seed (optional, 0 = random)", value=0, step=1)

    if st.button("Generate Art (DCGAN)"):
        if seed != 0:
            tf.random.set_seed(int(seed))
        noise = tf.random.normal([num_images, 100])
        generated = generator(noise, training=False).numpy()
        generated = (generated + 1) / 2.0
        st.session_state.generated_images = list(generated)
        st.session_state.generated_caption = "DCGAN Generated Art"

    # Show saved images if exist
    if st.session_state.generated_images:
        st.write("### Generated Images:")
        cols = st.columns(len(st.session_state.generated_images))
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                plt.imshow(np.clip(img, 0, 1))
                plt.axis("off")
                st.pyplot(plt)
                plt.clf()

                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label=f"📥 Download {i+1}",
                    data=byte_im,
                    file_name=f"dcgan_image_{i+1}.png",
                    mime="image/png",
                    key=f"dcgan_dl_{i}"  # unique key
                )

# ===============================
# 4️⃣ CGAN Section
# ===============================
elif model_choice == "CGAN":
    st.subheader("Conditional GAN (Anime Faces)")
    cgan_path = st.text_input("CGAN generator path", value="generator_CGAN.h5")
    with st.spinner("Loading CGAN generator..."):
        generator = load_generator(cgan_path)

    # Hair & Eye options from your CGAN training
    hair2idx = {
        'aqua': 0, 'gray': 1, 'green': 2, 'orange': 3, 'red': 4, 'white': 5,
        'black': 6, 'blonde': 7, 'blue': 8, 'brown': 9, 'pink': 10, 'purple': 11
    }
    eye2idx = {
        'aqua': 0, 'black': 1, 'blue': 2, 'brown': 3, 'green': 4, 'orange': 5,
        'pink': 6, 'purple': 7, 'red': 8, 'yellow': 9
    }
    num_hairs = len(hair2idx)
    num_eyes = len(eye2idx)
    noise_dim = 100

    hair_choice = st.selectbox("Select Hair Color:", list(hair2idx.keys()))
    eye_choice = st.selectbox("Select Eye Color:", list(eye2idx.keys()))

    def make_condition(hair, eye):
        cond = np.zeros(num_hairs + num_eyes, dtype=np.float32)
        cond[hair2idx[hair]] = 1
        cond[num_hairs + eye2idx[eye]] = 1
        return cond.reshape(1, -1)

    def generate_cgan_image(hair, eye):
        noise = np.random.normal(0, 1, (1, noise_dim))
        cond = make_condition(hair, eye)
        img = generator([noise, cond], training=False).numpy()
        img = (img[0] + 1) / 2.0
        img_small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        return img_small

    if st.button("Generate Art (CGAN)"):
        img = generate_cgan_image(hair_choice, eye_choice)
        st.session_state.generated_images = [img]
        st.session_state.generated_caption = f"{hair_choice} hair + {eye_choice} eyes"

    # Show saved image if exist
    if st.session_state.generated_images:
        img = st.session_state.generated_images[0]
        st.image(img, caption=st.session_state.generated_caption)

        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name=f"{hair_choice}_{eye_choice}.png",
            mime="image/png",
            key="cgan_dl"
        )
