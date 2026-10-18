"""URL-text feature extraction and readable risk signals."""
from __future__ import annotations
import re
from urllib.parse import urlparse

FEATURE_NAMES=["url_length","domain_length","path_length","dot_count","subdomain_count","hyphen_count","digit_count","at_symbol","has_ip_address","https","query_count","encoded_character_count","suspicious_keyword_count"]
SUSPICIOUS_WORDS={"login","verify","secure","account","update","confirm","wallet","signin","password","invoice","payment","unlock","recover"}

def normalise_url(value:str)->str:
    value=value.strip(); return value if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://",value) else "https://"+value

def extract_features(url:str)->list[int]:
    parsed=urlparse(normalise_url(url)); domain=(parsed.hostname or "").lower(); words=re.findall(r"[a-z]+",f"{domain} {parsed.path} {parsed.query}".lower())
    ip=bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}",domain))
    return [len(url),len(domain),len(parsed.path),url.count("."),max(0,domain.count(".")-1),url.count("-"),sum(c.isdigit() for c in url),int("@" in url),int(ip),int(parsed.scheme.lower()=="https"),url.count("?"),url.lower().count("%"),sum(w in SUSPICIOUS_WORDS for w in words)]

def risk_signals(url:str,values:list[int])->list[dict]:
    d=dict(zip(FEATURE_NAMES,values)); parsed=urlparse(url); domain=(parsed.hostname or "").lower(); out=[]
    def add(level,weight,title,detail): out.append({"level":level,"weight":weight,"title":title,"detail":detail})
    if d["has_ip_address"]: add("high",.45,"IP address used as the destination","Raw IP addresses make site ownership harder to verify.")
    if d["at_symbol"]: add("high",.40,"Contains an @ character","Everything before @ can be deceptive; browsers navigate to the host after it.")
    if d["subdomain_count"]>=3: add("high",.28,"Deep subdomain structure","Several subdomains can disguise the registered domain at the end of the hostname.")
    elif d["subdomain_count"]>=2: add("medium",.15,"Multiple subdomains","Read the domain from right to left to identify who controls it.")
    if "xn--" in domain or any(ord(c)>127 for c in domain): add("high",.35,"Internationalised domain encoding","Punycode or Unicode look-alikes can imitate familiar names.")
    if d["url_length"]>=100: add("medium",.18,"Unusually long address","Long URLs can conceal destination and parameters.")
    if d["hyphen_count"]>=2: add("medium",.14,"Multiple hyphens","Hyphenated look-alike domains are common in impersonation attempts, though not proof alone.")
    if d["encoded_character_count"]>=2: add("medium",.18,"Encoded URL characters","Encoding can make a destination or path harder to read.")
    if d["suspicious_keyword_count"]>=2: add("medium",.22,"Credential or urgency language","Several terms such as login, verify, account, or payment deserve scrutiny when unexpected.")
    elif d["suspicious_keyword_count"]==1: add("low",.07,"Sensitive-action wording","A single account or sign-in term should match the expected domain.")
    if parsed.scheme.lower()=="http": add("medium",.20,"No HTTPS encryption","HTTP does not protect information in transit; do not submit passwords or payment details.")
    if not out: add("low",0,"No major URL-pattern anomalies","No strong structural warning signs were found. This does not establish safety.")
    return sorted(out,key=lambda item:item["weight"],reverse=True)
