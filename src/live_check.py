"""Conservative metadata-only URL inspection for local PhishLens use."""
from __future__ import annotations
import ipaddress, socket
from urllib.parse import urlparse
import requests

def _public(host):
    try:
        addresses={row[4][0] for row in socket.getaddrinfo(host,None)}
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses): return False,"For safety, live checks do not contact private or reserved addresses."
        return True,", ".join(sorted(addresses))
    except socket.gaierror: return False,"The hostname could not be resolved."

def inspect_url(url):
    parsed=urlparse(url)
    if parsed.scheme not in {"http","https"} or not parsed.hostname: return {"ok":False,"summary":"Only public HTTP(S) URLs can be checked."}
    allowed,ips=_public(parsed.hostname)
    if not allowed: return {"ok":False,"summary":ips}
    try:
        headers={"User-Agent":"PhishLens/1.0 URL metadata checker"}
        response=requests.head(url,headers=headers,timeout=(3,6),allow_redirects=False)
        if response.status_code in {405,501}: response=requests.get(url,headers=headers,timeout=(3,6),allow_redirects=False,stream=True)
        details={"status":response.status_code,"resolved_ips":ips,"server":response.headers.get("Server","Not disclosed"),"redirect_target":response.headers.get("Location","No redirect in this response")}; response.close()
        note=" Inspect the target before following it." if 300<=details["status"]<400 else ""
        return {"ok":True,"summary":f"The server responded with HTTP {details['status']}.{note}","details":details}
    except requests.exceptions.SSLError: return {"ok":False,"summary":"The site’s TLS certificate could not be verified."}
    except requests.exceptions.Timeout: return {"ok":False,"summary":"The site did not respond within six seconds."}
    except requests.RequestException as exc: return {"ok":False,"summary":f"Live check could not complete: {exc.__class__.__name__}."}
