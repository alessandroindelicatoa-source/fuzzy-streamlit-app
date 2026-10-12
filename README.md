# Fuzzy Quadrant Mapper: Fuzzy Hybrid TOPSIS and ECO-Extended Apostle Model

Streamlit app that computes two latent variables with Fuzzy Hybrid TOPSIS, classifies respondents with the
Classic Apostle Model and the Fuzzy Clustering ECO-Extended Apostle Model, and estimates conditional
probability ratios with bootstrap confidence intervals.

The method follows Section 2 of Indelicato, A., Moral-Cuadra, S., Orgaz-Agüera, F. & Martín, J. C. (2026).
How well are environmental attitudes and sustainable transport alternatives aligned in Santiago de los
Caballeros (Dominican Republic). *International Journal of Sustainable Transportation*, 20(1), 1–15.
https://doi.org/10.1080/15568318.2025.2557324

## How to run
```bash
pip install -r requirements.txt
streamlit run app_fuzzy.py
```

## Authentication
If `.streamlit/secrets.toml` contains a `[passwords]` table, the app asks for a username and password.
Without it, the app is open.

## Citation
Indelicato, A. & Martín, J. C. (2025). *Fuzzy Quadrant Mapper: An Interactive App for Fuzzy Hybrid TOPSIS
and ECO-Extended Apostle Models*.
