# PrivacyShield: Adversarial Image Cloaking

## Abstract

**PrivacyShield** is a web-based application designed to protect user photos from being harvested and recognized by unauthorized Facial Recognition Systems (FRS). 

By applying **Adversarial Perturbations** (specifically PGD with Expectation Over Transformation), the tool modifies the pixel values of an image in a way that is nearly invisible to the human eye but completely disrupts the mathematical features used by AI models. This effectively "cloaks" the user's identity, preventing algorithms from matching the photo to their biometric profile.

## Key Features

* **Adversarial Cloaking:** Uses the **Projected Gradient Descent (PGD)** method to generate robust adversarial examples.
* **Screenshot Resistance:** Implements **Expectation Over Transformation (EOT)** to ensure protection survives resizing, compression, and digital screenshots.
* **Live Verification:** Integrated "Judge" model (InceptionResnetV1) to instantly verify if the protection was successful.
* **Adjustable Privacy:** User-controlled "Noise Level" slider to balance visual quality vs. protection strength.
* **Zero-Knowledge UI:** Simple drag-and-drop interface built with Streamlit—no coding required for the end-user.
