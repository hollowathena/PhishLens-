# PhishLens

A polished Streamlit prototype for URL risk assessment. It combines a lightweight ML ensemble with explainable URL-pattern signals, a metadata-only live check, and an explicit user-controlled option to open a destination.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`).

## Train on your study dataset

Once you have a documented dataset, train fresh models with a CSV containing `url,label`:

```powershell
python train_model.py path\to\labelled_urls.csv
```

It removes duplicate URLs before making a stratified 80/20 train/test split, prints held-out metrics, and saves a local model bundle in `models/`. Keep a copy of the source, licence, cleaning decisions, and test-set protocol for your own study records.

## Dataset safety and research use

The packaged training and survey records are fabricated, inert `example.*` / documentation IP examples. They make the interface runnable but **must not be used as research evidence**. Before your evaluation, replace them with a documented, licensed labelled URL source; remove duplicates before a stratified split; and reserve the test set exclusively for final testing.

The analyser does not automatically visit a submitted URL. Its optional live check blocks private and reserved network addresses, requests response metadata only, never follows redirects, and never executes page content. Opening a site is a separate, user-confirmed action in a new browser tab.

PhishLens cannot prove a site is safe and should not be used as a security control.
