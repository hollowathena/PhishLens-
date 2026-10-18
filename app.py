"""PhishLens — explainable URL risk assessment with optional live checks."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.demo_data import demo_urls
from src.features import FEATURE_NAMES, extract_features, normalise_url, risk_signals
from src.live_check import inspect_url

st.set_page_config(page_title="PhishLens | URL Risk Intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp{background:radial-gradient(circle at 12% -8%,#25345f 0,transparent 30%),radial-gradient(circle at 94% 8%,#123e50 0,transparent 25%),#080d19;color:#e7edf8;font-family:Manrope,system-ui,sans-serif}.block-container{max-width:1180px;padding-top:2.25rem;padding-bottom:4rem}.hero{position:relative;overflow:hidden;background:linear-gradient(116deg,#101a35 0%,#192a55 55%,#0f4b55 130%);color:#fff;border:1px solid #415b8c99;border-radius:24px;padding:2.5rem 2.7rem;margin-bottom:1.65rem;box-shadow:0 25px 65px #00000055}.hero:after{content:'◈';position:absolute;right:2.2rem;top:-1.35rem;font-size:9rem;color:#8ce3dc12;line-height:1}.hero h1{position:relative;margin:.1rem 0 0;font-size:2.65rem;letter-spacing:-.06em}.hero p{position:relative;margin:.6rem 0 0;color:#bbcae8;font-size:1.02rem}.eyebrow{position:relative;color:#70e5cf;font-size:.72rem;font-weight:800;letter-spacing:.16em}.panel{background:linear-gradient(145deg,#121b2d,#0d1424);border:1px solid #2c3b58;border-radius:18px;padding:1.35rem 1.5rem;box-shadow:0 14px 30px #00000028;margin:.45rem 0 1rem}.risk-high{border-left:5px solid #f05264;box-shadow:inset 0 0 35px #f0526412}.risk-moderate{border-left:5px solid #f4b740;box-shadow:inset 0 0 35px #f4b74012}.risk-low{border-left:5px solid #49d6a4;box-shadow:inset 0 0 35px #49d6a412}.risk-label{font-size:.71rem;letter-spacing:.13em;font-weight:800;text-transform:uppercase;color:#8495b6}.risk-value{font-size:2.15rem;font-weight:800;color:#f3f6ff;margin:.15rem 0}.risk-detail{color:#aab8d1;font-size:.9rem}.signal{display:flex;gap:.75rem;padding:.82rem 0;border-bottom:1px solid #28364f}.signal:last-child{border:0}.signal strong{display:block;color:#e8efff;margin-bottom:.12rem}.signal span{color:#9cabca;font-size:.92rem}label,p,li{color:#c3cee2!important}div[data-testid='stTextInput'] input{background:#101a2b!important;color:#eef4ff!important;border:1px solid #405276!important;border-radius:12px!important;padding:.8rem 1rem!important;box-shadow:inset 0 1px #ffffff08}div[data-testid='stTextInput'] input:focus{border-color:#5fd6c5!important;box-shadow:0 0 0 3px #5fd6c522!important}button[kind='primary'],button[kind='secondary']{border-radius:11px!important;border:1px solid #3c547b!important;background:linear-gradient(135deg,#1c3a72,#176b72)!important;color:#f5fbff!important;font-weight:700!important;transition:transform .18s ease,filter .18s ease!important}button[kind='primary']:hover,button[kind='secondary']:hover{transform:translateY(-1px);filter:brightness(1.15)}div[data-baseweb='tab-list']{gap:.45rem;border-bottom:1px solid #26344d;padding-bottom:.35rem}button[data-baseweb='tab']{color:#94a5c2!important;border-radius:9px!important;padding:.45rem .9rem!important}button[aria-selected='true']{color:#72e4d1!important;background:#163747!important}.stAlert{background:#111c2f!important;border:1px solid #314564!important;color:#cbd8ed!important;border-radius:14px!important}div[data-testid='stMetric']{background:#101a2b;border:1px solid #2c3d5a;border-radius:14px;padding:1rem}div[data-testid='stDataFrame']{border:1px solid #293955;border-radius:12px;overflow:hidden}.stMarkdown code{background:#17243b;color:#9af1da;padding:.15rem .35rem;border-radius:5px}.stCaption{color:#8fa1c0!important}</style>""", unsafe_allow_html=True)
st.markdown("<style>.block-container,.stMainBlockContainer{padding-top:3.4rem!important}.hero{margin-top:.15rem!important}</style>", unsafe_allow_html=True)
MODEL_BUNDLE = Path("models/model_bundle.joblib")

@st.cache_resource
def load_models():
    if MODEL_BUNDLE.exists():
        bundle=joblib.load(MODEL_BUNDLE)
        if {"logistic","forest"} <= set(bundle): return bundle["logistic"],bundle["forest"],True
    data=demo_urls().drop_duplicates("url"); X=pd.DataFrame([extract_features(u) for u in data.url],columns=FEATURE_NAMES); y=data.label.astype(int)
    X_train,_,y_train,_=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    return (LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42).fit(X_train,y_train),RandomForestClassifier(n_estimators=350,max_depth=10,min_samples_leaf=2,class_weight="balanced",random_state=42).fit(X_train,y_train),False)

def assess(url,logistic,forest):
    values=extract_features(url); row=pd.DataFrame([values],columns=FEATURE_NAMES)
    model=(float(logistic.predict_proba(row)[0,1])+float(forest.predict_proba(row)[0,1]))/2
    signals=risk_signals(url,values); structural=min(1,sum(item["weight"] for item in signals))
    return model*.70+structural*.30,model,signals

logistic,forest,research_model=load_models()
st.markdown("""<section class="hero"><div class="eyebrow">URL RISK INTELLIGENCE</div><h1>PhishLens</h1><p>Understand destination patterns, inspect warning signals, and decide whether to visit with more context.</p></section>""",unsafe_allow_html=True)
analysis_tab,guide_tab=st.tabs(["URL analyser","How it works"])
with analysis_tab:
    submitted=st.text_input("Paste a URL or domain",placeholder="https://portal.example.edu/login")
    st.caption("The address is analysed as text first. Live checks and visiting are optional actions you control.")
    if submitted:
        url=normalise_url(submitted); parsed=urlparse(url)
        if parsed.scheme not in {"http","https"} or not parsed.hostname or "." not in parsed.hostname:
            st.error("Enter a complete public HTTP(S) address, such as `https://example.edu/login`.")
        else:
            score,model,signals=assess(url,logistic,forest)
            level,klass,action=("High","high","Do not sign in or download anything. Verify through the organisation’s official site or app.") if score>=.68 else (("Caution","moderate","Pause and independently verify the registered domain before interacting.") if score>=.38 else ("Lower","low","No strong structural warnings were found. Still verify unexpected links independently."))
            a,b=st.columns([.75,1.25])
            with a: st.markdown(f'<div class="panel risk-{klass}"><div class="risk-label">URL pattern assessment</div><div class="risk-value">{level} risk</div><div class="risk-detail">Risk indicator: {score:.0%}</div></div>',unsafe_allow_html=True)
            with b: st.markdown(f'<div class="panel"><div class="risk-label">Recommended next step</div><div class="risk-value" style="font-size:1.25rem">{action}</div><div class="risk-detail">This is an indicator, not proof of safety or malicious intent.</div></div>',unsafe_allow_html=True)
            left,right=st.columns([1.45,1])
            with left:
                st.markdown("#### Why this URL was assessed this way")
                for s in signals: st.markdown(f'<div class="signal"><div>●</div><div><strong>{s["title"]}</strong><span>{s["detail"]}</span></div></div>',unsafe_allow_html=True)
                st.caption(f"Ensemble indicator: {model:.0%}. Structural cross-check: {min(1,sum(s['weight'] for s in signals)):.0%}.")
            with right:
                st.markdown("#### URL at a glance")
                st.metric("Destination hostname",parsed.hostname)
                st.dataframe(pd.DataFrame({"Attribute":["Protocol","Characters","Subdomains","Query parameters"],"Value":[parsed.scheme.upper(),len(url),max(0,parsed.hostname.count('.')-1),url.count('?')]}),hide_index=True,use_container_width=True)
            st.divider(); st.markdown("#### Optional live check and visit")
            st.info("Live checking contacts the destination once to read response headers only. It blocks local/private network addresses and does not follow redirects or execute page content.")
            c,d=st.columns(2)
            with c:
                if st.button("Run safe live check",use_container_width=True):
                    with st.spinner("Checking response metadata…"): result=inspect_url(url)
                    (st.success if result["ok"] else st.warning)(result["summary"])
                    if result.get("details"): st.json(result["details"],expanded=False)
            with d:
                consent=st.checkbox("I understand this opens the destination in a new browser tab.")
                if consent: st.link_button("Open destination in new tab ↗",url,use_container_width=True)
                else: st.button("Open destination in new tab ↗",disabled=True,use_container_width=True)
with guide_tab:
    st.markdown("### Designed for informed decisions, not false certainty")
    st.markdown("PhishLens combines two URL-text classifiers with structural checks for look-alike subdomains, raw IP destinations, encoded characters, misleading `@` notation, insecure HTTP, and credential-themed wording.")
    st.markdown("The live check requests response metadata only. It does not render a page, run scripts, submit forms, download files, or follow redirects. Opening a URL is a separate, user-confirmed action.")
    (st.success("A locally trained model bundle is active.") if research_model else st.warning("The default model uses safe fabricated examples. Train `train_model.py` on a documented dataset before making performance claims."))
    st.markdown("#### Verification checklist\n1. Find the organisation independently via a saved bookmark or official search result.\n2. Compare the exact registered domain, not only the words at the start.\n3. Treat unexpected credential, payment, or urgent-action requests with extra caution.\n4. Use browser and endpoint protection for real-time security controls.")
