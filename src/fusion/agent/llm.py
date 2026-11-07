"""Client minimal pour les API de chat « compatibles OpenAI » (Groq, OpenRouter, Ollama, Gemini en mode compatible...).
Bibliothèque standard seulement (pas de dépendance). Gère les limites de débit (HTTP 429) par reprise avec attente.
Les URL de base ci-dessous viennent de ma connaissance des services, pas d'un test ici : à confirmer dans leur documentation."""
import json
import os
import re
import time
import urllib.error
import urllib.request

BASES = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def strip_reasoning(txt):
    """Retire les blocs de raisonnement <think>...</think> que certains modèles ajoutent avant la réponse."""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()


def make_chat(base_url, model, api_key=None, system="", sleep=0.0, retries=5, timeout=120, max_tokens=700):
    """Retourne call(texte) -> réponse (str). `sleep` : pause entre deux appels (limite de tokens par minute)."""
    url = base_url.rstrip("/") + "/chat/completions"
    last = [0.0]

    def call(text):
        wait = sleep - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]}
        # Certains services (Groq derrière Cloudflare : « 403 error code: 1010 ») rejettent le User-Agent par défaut de Python.
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "carla-lidar-camera-fusion-agent-eval/1.0"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                last[0] = time.time()
                return strip_reasoning(data["choices"][0]["message"].get("content") or "")
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < retries:
                    ra = e.headers.get("Retry-After")
                    time.sleep(min(float(ra), 60) if ra and ra.replace(".", "").isdigit() else min(2 ** attempt * 2, 60))
                    continue
                raise RuntimeError(f"HTTP {e.code} : {e.read().decode(errors='replace')[:300]}") from e
        raise RuntimeError("trop de tentatives")
    return call


def key_from_env(*names):
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None
