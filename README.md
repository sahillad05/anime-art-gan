# anime-art-gan
Generating Anime Art using GANs... This project implements DCGAN and CGAN to generate anime faces... DCGAN creates random faces, while CGAN generates faces conditioned on hair and eye color... Both models are trained on the Anime Face Dataset and integrated into a Streamlit app for real-time interactive generation...

# Generating Anime Art using GANs  

This project demonstrates the power of **Generative Adversarial Networks (GANs)** to create anime faces. Two models are used:  

- DCGAN – Generates random anime faces.  
- CGAN – Generates anime faces conditioned on **hair color** and **eye color**.  

Both models are integrated into a **Streamlit app** for an interactive user experience.  

---

## 📂 Datasets  
- DCGAN Dataset → [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)  
- CGAN Dataset → [Anime Face with Hair & Eye Color Tags](https://www.kaggle.com/datasets/mnkbiswas/anime-face-with-eye-and-hair-color-tagged)  

---

## Features in Streamlit App  
- Model Selection → Choose between **DCGAN** and **CGAN**.  
- DCGAN Mode →  
  - Input: Number of images to generate.  
  - Seed option (0 = random).  
  - Output: Generated anime faces.  
- CGAN Mode →  
  - Select **Hair Color** & **Eye Color**.  
  - Output: Anime face generated based on chosen labels.  
- Download Option → Save generated images locally.  

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/sahillad05/anime-art-gan.git
cd anime-art-gan

```

Run the Streamlit app

```bash
streamlit run app.py
```
