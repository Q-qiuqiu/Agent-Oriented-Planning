import requests


def chat_completions_url(base_url):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def auth_header(api_key):
    if not api_key:
        return {}
    if api_key.lower().startswith("bearer "):
        return {"Authorization": api_key}
    return {"Authorization": f"Bearer {api_key}"}


def run_chat_completion(model, prompt, api_url, api_key="", timeout=120, temperature=0.0, system_prompt=None):
    headers = {"Content-Type": "application/json"}
    headers.update(auth_header(api_key))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(
        chat_completions_url(api_url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
