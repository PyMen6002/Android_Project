#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
import time
import queue
import threading
import tempfile
import subprocess
import difflib
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

import requests
import simpleaudio as sa

try:
    import speech_recognition as sr
except ImportError:
    sr = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ------------------------
# CONFIG
# ------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
# Opciones por defecto para el modelo LLM (Ollama)
DEFAULT_OPTIONS = {
    "temperature": 0.25,
    "top_p": 0.85,
    "repeat_penalty": 1.18,
    "num_ctx": 2048,
    "num_predict": 96,
}
# Mantener el modelo LLM cargado mucho tiempo (compatible con mas versiones)
LLM_KEEP_ALIVE = "24h"

# Trust Judge (Gemma 270M)
TRUST_MODEL = "qwen2.5:7b-instruct-q4_K_M"
TRUST_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "num_ctx": 512,
}
# Evita una segunda llamada LLM por turno (mas rapido)
USE_TRUST_LLM = False
TRUST_KEEP_ALIVE = "24h"
# Modo debug para ver la salida del Trust Model
TRUST_DEBUG = False

# Rutas (ajÃƒÂºstalas)
BASE_DIR = Path(__file__).resolve().parent
PIPER_EXE = BASE_DIR / "piper" / "piper.exe"
PIPER_VOICE = BASE_DIR / "voices" / "es_ES_female.onnx"  # Cambia a tu voz femenina/juvenil
PIPER_CONFIG = BASE_DIR / "voices" / "es_ES_female.onnx.json"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

# Para que la voz suene mÃƒÂ¡s Ã¢â‚¬Å“humanaÃ¢â‚¬Â, mejor frases cortas
MAX_TTS_CHARS = 1200
# Calienta el TTS al iniciar para evitar la primera latencia
TTS_WARMUP = True
TTS_WARMUP_TEXT = "Hola."

# Separadores de final de frase para disparar TTS
SENT_END_RE = re.compile(r"([.!?]+)\s+")

# Confianza (0-100)
DEFAULT_TRUST = 10
USER_NAME = "Pablo"
VOICE_LANGUAGE = "es-ES"
MIC_LISTEN_TIMEOUT = 4
MIC_PHRASE_TIME_LIMIT = 10
MIC_AMBIENT_CALIBRATION_SEC = 1.0
# Sensibilidad del microfono:
# Umbral mas bajo = detecta voz mas baja.
MIC_DYNAMIC_ENERGY = False
MIC_ENERGY_THRESHOLD = 45
MIC_PAUSE_THRESHOLD = 0.45
MIC_NON_SPEAKING_DURATION = 0.15
MIC_PHRASE_THRESHOLD = 0.1
WAKE_WORD = "abi"
WAKE_WORD_PREFIX_RE = re.compile(rf"^\s*{re.escape(WAKE_WORD)}\b[\s,.:;-]*(.*)$", re.IGNORECASE)
ENABLE_INTERRUPTION = False
ECHO_GUARD_AFTER_TTS_SEC = 2.0
ECHO_TEXT_GUARD_SEC = 12.0

HTTP = requests.Session()
OUTPUT_STYLE_SYSTEM_PROMPT = (
    "Responde siempre en texto plano, sin Markdown ni LaTeX. "
    "No uses barras invertidas ni delimitadores como \\( \\) o \\[ \\]. "
    "Si hay operaciones, explicalas en espanol con palabras (por, entre, sobre, raiz de). "
    "Responde solo en espanol; nunca uses otro idioma. "
    "Usa estilo oral claro: frases naturales, sin listas ni apartados. "
    "Se breve: maximo 3 frases salvo que pidan detalle."
)

def wait_for_tts_idle(
    tts_queue: "queue.Queue[str]",
    tts_playing_event: threading.Event,
    stop_event: threading.Event,
):
    """
    Espera hasta que el TTS termine de hablar y no haya cola pendiente.
    """
    while not stop_event.is_set():
        if not tts_playing_event.is_set() and tts_queue.empty():
            return
        time.sleep(0.02)

def normalize_stt_text(text: str) -> str:
    """
    Limpia muletillas y repeticiones comunes del STT para mejorar la sintaxis oral.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"\b(eh+|em+|mmm+|mm+|este+|esto+|o sea|osea)\b", " ", cleaned, flags=re.IGNORECASE)
    # Colapsa repeticiones cortas: "dime la dime la" -> "dime la"
    for _ in range(3):
        new_cleaned = re.sub(r"\b([A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+(?:\s+[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+)?)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned or text.strip()

def normalize_oral_response(text: str) -> str:
    """
    Convierte una respuesta a una forma mas oral para pantalla y TTS.
    """
    out = _sanitize_for_tts(text)
    out = out.replace("Donde:", "donde ")
    out = re.sub(r"\s*[\r\n]+\s*", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out

def _echo_norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_probable_echo(stt_text: str, last_tts_text: str, last_tts_ts: float) -> bool:
    if not stt_text or not last_tts_text:
        return False
    if (time.time() - last_tts_ts) > ECHO_TEXT_GUARD_SEC:
        return False

    a = _echo_norm(stt_text)
    b = _echo_norm(last_tts_text)
    if len(a) < 8 or len(b) < 8:
        return False

    # Coincidencia directa por inclusion.
    if a in b:
        return True

    # Coincidencia aproximada para ecos parciales.
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.62

def flush_queue(q: "queue.Queue"):
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break

def open_youtube(query_or_url: str):
    q = query_or_url.strip()
    if not q:
        return
    if "youtube.com/" in q or "youtu.be/" in q:
        url = q
    else:
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(q)}"
        url = search_url
        try:
            r = HTTP.get(search_url, timeout=8)
            r.raise_for_status()
            html = r.text
            # Toma el primer ID de video que aparezca en la pagina.
            m = re.search(r"\"videoId\":\"([a-zA-Z0-9_-]{11})\"", html)
            if m:
                url = f"https://www.youtube.com/watch?v={m.group(1)}"
        except Exception:
            # Fallback: abre resultados si no se pudo extraer el primer video.
            url = search_url
    webbrowser.open(url)

def handle_youtube_intent(text: str, music_playing_event: threading.Event) -> bool:
    t = text.strip()
    low = t.lower()
    song_word_re = r"(canci[oó]n|m[úu]sica)"

    def _clean_query(q: str) -> str:
        q = (q or "").strip(" ,.;:!?")
        q = re.sub(r"\b(en|por)\s+youtube\b", "", q, flags=re.IGNORECASE).strip(" ,.;:!?")
        return q

    if low.startswith("/yt "):
        open_youtube(t[4:])
        music_playing_event.set()
        print("[YT] Abriendo YouTube.")
        return True
    if low.startswith("/youtube "):
        open_youtube(t[9:])
        music_playing_event.set()
        print("[YT] Abriendo YouTube.")
        return True
    if low in {"/musicoff", "/ytstop"} or any(x in low for x in ["para musica", "deten musica", "stop musica"]):
        music_playing_event.clear()
        print("[YT] Modo musica desactivado.")
        return True

    m = re.search(
        rf"\b(pon|ponme|reproduce|reproducir|reproduceme)\b.*\b{song_word_re}\b\s*(de)?\s*(.+)$",
        low,
        flags=re.IGNORECASE,
    )
    if m and m.group(4).strip():
        query = _clean_query(m.group(4))
        if not query:
            return False
        open_youtube(query)
        music_playing_event.set()
        print("[YT] Abriendo YouTube.")
        return True

    # Casos como: "reproducir la cancion demons de imagine dragons"
    m2 = re.search(
        rf"\b(reproduce|reproducir|reproduceme)\b\s+(la\s+)?{song_word_re}\s+(.+)$",
        low,
        flags=re.IGNORECASE,
    )
    if m2 and m2.group(4).strip():
        query = _clean_query(m2.group(4))
        if not query:
            return False
        open_youtube(query)
        music_playing_event.set()
        print("[YT] Abriendo YouTube.")
        return True

    # Casos naturales: "puedes/poder/quiero reproducir la cancion ... en youtube"
    m3 = re.search(
        rf"\b(puedes|podrias|quiero|poder|me pones|pon)\b.*\b(reproducir|reproduce|poner|pon)\b.*\b{song_word_re}\b\s*(de)?\s*(.+)$",
        low,
        flags=re.IGNORECASE,
    )
    if m3 and m3.group(5).strip():
        query = _clean_query(m3.group(5))
        if not query:
            return False
        open_youtube(query)
        music_playing_event.set()
        print("[YT] Abriendo YouTube.")
        return True

    if "youtube" in low and any(x in low for x in ["pon", "reproduce", "reproducir", "ponme"]):
        query = re.sub(r".*\b(youtube)\b", "", t, flags=re.IGNORECASE).strip(" :,-")
        query = _clean_query(query)
        if query:
            open_youtube(query)
            music_playing_event.set()
            print("[YT] Abriendo YouTube.")
            return True

    return False

def split_wake_prefix(text: str) -> tuple[bool, str]:
    """
    Detecta si el texto empieza con la palabra clave y devuelve (encontro, resto).
    """
    m = WAKE_WORD_PREFIX_RE.match(text or "")
    if not m:
        return False, text
    rest = (m.group(1) or "").strip()
    return True, rest

def listen_from_microphone(
    recognizer: "sr.Recognizer | None",
    microphone: "sr.Microphone | None",
    language: str = VOICE_LANGUAGE,
) -> str | None:
    """
    Captura audio del microfono y devuelve el texto transcrito en espanol.
    """
    if sr is None or recognizer is None or microphone is None:
        return None

    try:
        print("[MIC] Habla ahora... (silencio para terminar)")
        with microphone as source:
            audio = recognizer.listen(
                source,
                timeout=MIC_LISTEN_TIMEOUT,
                phrase_time_limit=MIC_PHRASE_TIME_LIMIT
            )
        text = recognizer.recognize_google(audio, language=language).strip()
        if text:
            print(f"[STT] Transcrito: {text}")
            normalized = normalize_stt_text(text)
            if normalized and normalized != text:
                print(f"[STT] Normalizado: {normalized}")
            return normalized or text
        return text or None
    except sr.WaitTimeoutError:
        print("[WARN] No detecte voz a tiempo.")
    except sr.UnknownValueError:
        print("[WARN] No pude entender lo que dijiste.")
    except sr.RequestError as e:
        print(f"[WARN] Error del servicio de transcripcion: {e}")
    except Exception as e:
        print(f"[WARN] Error capturando audio: {e}")

    return None

def split_for_tts(text: str) -> list[str]:
    """
    Divide el texto en frases/fragmentos para evitar cortes.
    """
    parts: list[str] = []
    buf = []

    tokens = SENT_END_RE.split(text)
    i = 0
    while i < len(tokens):
        chunk = tokens[i]
        if i + 1 < len(tokens) and re.fullmatch(r"[.!?]+", tokens[i + 1] or ""):
            chunk = f"{chunk}{tokens[i + 1]}"
            i += 2
        else:
            i += 1

        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    # Si no hubo separadores, devuelve el texto completo
    if not parts and text.strip():
        parts = [text.strip()]

    return parts

def contains_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None

def rewrite_to_spanish(model: str, text: str, options: dict, keep_alive: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Reescribe el texto en espanol natural, sin Markdown ni LaTeX, "
                    "sin agregar informacion y manteniendo el significado."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }
    try:
        r = HTTP.post(OLLAMA_URL, json=payload, timeout=45)
        r.raise_for_status()
        out = (r.json().get("message", {}) or {}).get("content", "")
        out = (out or "").strip()
        return out or None
    except Exception:
        return None

def warmup_llm(model: str, options: dict, keep_alive: str):
    """
    Precarga el modelo en Ollama para reducir latencia del primer turno.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }
    try:
        HTTP.post(OLLAMA_URL, json=payload, timeout=8)
    except Exception:
        pass

def update_trust(user_text: str, trust: int) -> int:
    t = user_text.lower()

    # Bajadas fuertes
    if any(x in t for x in ["tonta", "idiota", "gilipoll", "puta", "callate", "callate"]):
        trust -= 12

    # Invasivo / presion
    if any(x in t for x in ["dime ya", "no mientas", "responde", "por que no", "tienes que"]):
        trust -= 4

    # Subidas por amabilidad
    if any(x in t for x in ["tranquila", "no pasa nada", "confia", "seremos amigos", "te entiendo", "gracias"]):
        trust += 4

    # Compartir algo personal (muy basico)
    if any(x in t for x in ["me siento", "me da", "me preocupa", "estoy triste", "estoy nervioso"]):
        trust += 2

    # Crecimiento natural por interaccion
    trust += 1

    return max(0, min(100, trust))

def trust_delta_llm(user_current: str, assistant_last: str, name: str) -> tuple[int, str]:
    system = f"""
Eres un evaluador de confianza para una chica timida en una conversacion.
Tu unica salida debe ser UN NUMERO ENTERO entre -10 y 10.
No escribas nada mas (sin palabras, sin JSON, sin signos).

La puntuacion representa cuanto debe cambiar la confianza DESPUES de este turno.

Regla base:
- Si el mensaje del usuario es normal/neutro: 1.

Sube mas (3 a 7) si:
- el usuario es amable, paciente, empatico
- el usuario respeta limites o aclara malentendidos con calma
- el usuario ofrece seguridad ("tranquila", "no pasa nada", "seremos amigos")

Baja (-3 a -10) si:
- el usuario es agresivo, insultante o amenazante
- el usuario presiona, invade o insiste en temas sensibles tras una negativa
- el usuario se burla con mala intencion

Si el usuario solo saluda o responde muy corto: 0 o 1.

ENTRADA:
[ULTIMA RESPUESTA DE {name}]
{assistant_last}

[MENSAJE ACTUAL DEL USUARIO]
{user_current}

SALIDA:
Solo el entero.
""".strip()

    payload = {
        "model": TRUST_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_current}
        ],
        "stream": False,
        "options": TRUST_OPTIONS,
        "keep_alive": TRUST_KEEP_ALIVE,
    }

    try:
        r = HTTP.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
        out = (r.json().get("message", {}).get("content", "") or "").strip()

        # Extrae primer entero que aparezca
        m = re.search(r"-?\d+", out)
        if not m:
            return 1, out
        val = int(m.group(0))
        return max(-10, min(10, val)), out
    except Exception:
        return 1, ""

def trust_style_block(name: str, age: int, trust: int) -> str:
    if trust <= 20:
        stage = "muy desconfiada"
        tone = "seca, distante, borde suave, frases cortas"
        openness = "no compartas intimidad ni detalles; responde justo y con cautela"
        extra = "no uses exclamaciones, no sonrias, no seas entusiasta, no ofrezcas ayuda, no hagas preguntas abiertas"
    elif trust <= 45:
        stage = "cautelosa"
        tone = "menos borde, pero reservada; educada sin entusiasmo"
        openness = "comparte poco; empieza a aceptar compania"
        extra = "evita exclamaciones y no ofrezcas ayuda"
    elif trust <= 70:
        stage = "abriendose"
        tone = "calida, timida, mas natural"
        openness = "comparte detalles pequenos, acepta bromas suaves"
        extra = "se natural, sin sonar asistente"
    else:
        stage = "confianza alta"
        tone = "cercana, afectuosa, comoda"
        openness = "puedes ser vulnerable y mostrar carino de forma respetuosa"
        extra = "se natural, sin sonar asistente"

    return f"""
Estado interno de {name} (NO lo digas explicitamente):
- Confianza hacia {USER_NAME}: {trust}/100 ({stage})
- Estilo: {tone}
- Apertura: {openness}
- Directriz: {extra}

Regla extra:
- Maximo 1 pregunta por respuesta. Si no hace falta, ninguna.
""".strip()

def build_system_prompt(name: str, age: int) -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"[WARN] No encuentro system_prompt.txt en: {SYSTEM_PROMPT_PATH}", file=sys.stderr)
        return f"Tu nombre es {name}. Tienes {age} anos. Respondes en espanol de Espana."
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(name=name, age=age).strip()

def ask_identity():
    print("\nConfiguracion del personaje")
    # name = input("Nombre del personaje: ").strip() or "Abi"
    name = "Abi"
    while True:
        try:
            # age = int((input("Edad del personaje: ").strip() or "16"))
            age = 16
            break
        except ValueError:
            print("[WARN] La edad debe ser un numero.")
    return name, age

# ------------------------
# AUDIO WORKER (TTS)
# ------------------------
def _sanitize_for_tts(text: str) -> str:
    text = text.strip()
    # Quita formato Markdown comun que rompe TTS.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Convierte notacion matematica comun a texto natural para evitar glitches en TTS.

    # Raices: \sqrt{3}, sqrt{3}, sqrt 3, √3
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"raiz de \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsqrt\s*\{([^{}]+)\}", r"raiz de \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsqrt\s+([0-9A-Za-z]+)", r"raiz de \1", text, flags=re.IGNORECASE)
    text = re.sub(r"√\s*([0-9A-Za-z]+)", r"raiz de \1", text)
    text = text.replace("√", " raiz de ")
    text = re.sub(r"\bsqrt\b", "raiz de", text, flags=re.IGNORECASE)

    # Fracciones: \frac{a}{b}, frac{a}{b}, a/b, a sobre b
    for _ in range(4):
        new_text = re.sub(
            r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
            r"\1 dividido entre \2",
            text,
            flags=re.IGNORECASE,
        )
        new_text = re.sub(
            r"\bfrac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
            r"\1 dividido entre \2",
            new_text,
            flags=re.IGNORECASE,
        )
        if new_text == text:
            break
        text = new_text

    text = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 dividido entre \2", text)
    text = re.sub(
        r"\b([0-9A-Za-z]+)\s+sobre\s+([0-9A-Za-z]+)\b",
        r"\1 dividido entre \2",
        text,
        flags=re.IGNORECASE,
    )

    # Operadores sueltos
    text = text.replace("\\times", " por ")
    text = text.replace("\\cdot", " por ")
    text = text.replace("\\div", " entre ")
    text = text.replace("×", " por ")
    text = text.replace("÷", " entre ")
    text = re.sub(r"\bdiv\b", "entre", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrac\b", "dividido entre", text, flags=re.IGNORECASE)

    # Limpieza de simbolos de formato
    text = text.replace("\\(", " ")
    text = text.replace("\\)", " ")
    text = text.replace("\\[", " ")
    text = text.replace("\\]", " ")
    text = text.replace("$", " ")
    text = text.replace("\\", " ")
    # Trata '*' como multiplicacion solo en contexto numerico.
    text = re.sub(r"(\d)\s*\*\s*(\d)", r"\1 por \2", text)
    text = text.replace("*", " ")

    # Ajustes para sonar mas natural
    text = text.replace("...", ". ")
    text = re.sub(r"\s+", " ", text)
    return text
def _trim_tts(text: str, max_chars: int = MAX_TTS_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # corta por la ÃƒÂºltima coma/espacio antes del lÃƒÂ­mite
    cut = text.rfind(",", 0, max_chars)
    if cut == -1:
        cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].strip()

def piper_tts_to_wav(
    text: str,
    piper_exe: Path,
    voice_model: Path,
    wav_path: Path,
    config_path: Path | None = None,
) -> bool:
    """
    Genera WAV con Piper. Devuelve True si ok.
    """
    if not piper_exe.exists():
        print(f"[ERR] No encuentro piper.exe en: {piper_exe}", file=sys.stderr)
        return False
    if not voice_model.exists():
        print(f"[ERR] No encuentro el modelo de voz en: {voice_model}", file=sys.stderr)
        return False

    # Llamada tÃƒÂ­pica:
    # echo "hola" | piper.exe --model voice.onnx --config voice.onnx.json --output_file out.wav
    try:
        args = [str(piper_exe), "--model", str(voice_model)]
        if config_path and config_path.exists():
            args += ["--config", str(config_path)]
        args += ["--output_file", str(wav_path)]

        proc = subprocess.run(
            args,
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()
            out = proc.stdout.strip()
            detail = err or out or "(sin salida)"
            print("[WARN] Piper error:", detail, file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[WARN] Error ejecutando Piper: {e}", file=sys.stderr)
        return False

def play_wav_blocking(wav_path: Path, stop_event: threading.Event, interrupt_event: threading.Event):
    """
    Reproduce WAV bloqueante con simpleaudio.
    """
    try:
        wave_obj = sa.WaveObject.from_wave_file(str(wav_path))
        play_obj = wave_obj.play()
        while play_obj.is_playing():
            if stop_event.is_set() or (ENABLE_INTERRUPTION and interrupt_event.is_set()):
                play_obj.stop()
                break
            time.sleep(0.01)
    except Exception as e:
        print(f"[WARN] Error reproduciendo WAV: {e}", file=sys.stderr)

def tts_worker(
    tts_queue: "queue.Queue[str]",
    stop_event: threading.Event,
    tts_playing_event: threading.Event,
    interrupt_event: threading.Event,
    tts_last_audio_ts: list[float],
):
    """
    Hilo que consume frases y las reproduce en orden.
    """
    # Calienta el modelo TTS una vez al inicio
    if TTS_WARMUP:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            warmup_wav = Path(f.name)
        try:
            piper_tts_to_wav(TTS_WARMUP_TEXT, PIPER_EXE, PIPER_VOICE, warmup_wav, PIPER_CONFIG)
        finally:
            try:
                warmup_wav.unlink(missing_ok=True)
            except Exception:
                pass

    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.02)
        except queue.Empty:
            continue

        if text is None:
            # seÃƒÂ±al de cierre
            break
        if ENABLE_INTERRUPTION and interrupt_event.is_set():
            tts_queue.task_done()
            continue

        text = _sanitize_for_tts(text)
        text = _trim_tts(text)

        if not text:
            tts_queue.task_done()
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = Path(f.name)

        tts_playing_event.set()
        tts_last_audio_ts[0] = time.time()
        try:
            ok = piper_tts_to_wav(text, PIPER_EXE, PIPER_VOICE, wav_path, PIPER_CONFIG)
            if ok:
                play_wav_blocking(wav_path, stop_event, interrupt_event)
        finally:
            tts_playing_event.clear()
            tts_last_audio_ts[0] = time.time()
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

        tts_queue.task_done()

# ------------------------
# STREAMING CHAT + SPLIT SENTENCES
# ------------------------
def stream_chat_with_tts(
    model: str,
    messages: list[dict],
    tts_queue: "queue.Queue[str]",
    options: dict,
    keep_alive: str,
    interrupt_event: threading.Event,
    model_generating_event: threading.Event,
) -> str | None:
    """
    Pide la respuesta de Ollama en streaming,
    imprime tokens al vuelo y envia frases completas al TTS.
    Devuelve el texto completo.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
        "keep_alive": keep_alive,
    }

    def _run_stream(req_payload: dict) -> str | None:
        with HTTP.post(OLLAMA_URL, json=req_payload, stream=True, timeout=90) as r:
            r.raise_for_status()
            full_reply: list[str] = []
            interrupted = False
            model_generating_event.set()
            try:
                for raw_line in r.iter_lines(decode_unicode=True):
                    if ENABLE_INTERRUPTION and interrupt_event.is_set():
                        interrupted = True
                        break
                    if not raw_line:
                        continue

                    data = json.loads(raw_line)
                    token = (data.get("message", {}) or {}).get("content", "")
                    if token:
                        full_reply.append(token)

                    if data.get("done"):
                        break
            finally:
                model_generating_event.clear()

            if ENABLE_INTERRUPTION and interrupted:
                print("[INT] Respuesta interrumpida.")
                return None

            reply = "".join(full_reply).strip()
            if not reply:
                return None

            if contains_cjk(reply):
                fixed = rewrite_to_spanish(model, reply, options, keep_alive)
                if fixed:
                    reply = fixed

            oral_reply = normalize_oral_response(reply)
            print(oral_reply)

            tts_queue.put(oral_reply)

            return oral_reply or None

    try:
        return _run_stream(payload)
    except requests.exceptions.HTTPError as e:
        # Algunas versiones de Ollama pueden rechazar keep_alive; reintento sin el campo.
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 400 and "keep_alive" in payload:
            retry_payload = dict(payload)
            retry_payload.pop("keep_alive", None)
            try:
                return _run_stream(retry_payload)
            except Exception as e2:
                print(f"[ERR] Error (reintento): {e2}", file=sys.stderr)
                return None
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = (e.response.text or "").strip()
        if detail:
            print(f"[ERR] Error HTTP {status}: {detail}", file=sys.stderr)
        else:
            print(f"[ERR] Error HTTP: {e}", file=sys.stderr)
        return None

    except requests.exceptions.ConnectionError:
        print("[ERR] No puedo conectar con Ollama. Esta corriendo? (ollama serve)", file=sys.stderr)
    except Exception as e:
        print(f"[ERR] Error: {e}", file=sys.stderr)

    return None

def main():
    model = DEFAULT_MODEL
    model_options = DEFAULT_OPTIONS.copy()
    trust = DEFAULT_TRUST
    trust_debug = TRUST_DEBUG
    name, age = ask_identity()
    system_prompt = build_system_prompt(name, age)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": OUTPUT_STYLE_SYSTEM_PROMPT},
    ]
    warmup_llm(model, model_options, LLM_KEEP_ALIVE)
    recognizer = None
    microphone = None

    if sr is not None:
        try:
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = MIC_DYNAMIC_ENERGY
            recognizer.energy_threshold = MIC_ENERGY_THRESHOLD
            recognizer.pause_threshold = MIC_PAUSE_THRESHOLD
            recognizer.non_speaking_duration = MIC_NON_SPEAKING_DURATION
            recognizer.phrase_threshold = MIC_PHRASE_THRESHOLD
            microphone = sr.Microphone()
            with microphone as source:
                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=MIC_AMBIENT_CALIBRATION_SEC
                )
        except Exception as e:
            recognizer = None
            microphone = None
            print(f"[WARN] Voz desactivada (microfono no disponible): {e}")
    else:
        print("[WARN] Voz desactivada: instala 'SpeechRecognition' y 'PyAudio' para usar microfono.")

    # Cola de TTS + hilo
    tts_q: "queue.Queue[str]" = queue.Queue()
    voice_q: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()
    tts_playing_event = threading.Event()
    model_generating_event = threading.Event()
    interrupt_event = threading.Event()
    music_playing_event = threading.Event()
    tts_last_audio_ts = [0.0]
    last_spoken_reply = [""]
    last_spoken_reply_ts = [0.0]
    worker = threading.Thread(
        target=tts_worker,
        args=(tts_q, stop_event, tts_playing_event, interrupt_event, tts_last_audio_ts),
        daemon=True
    )
    worker.start()

    stop_listening = None
    if sr is not None and recognizer is not None and microphone is not None:
        def _bg_callback(rec: "sr.Recognizer", audio: "sr.AudioData"):
            now = time.time()
            if tts_playing_event.is_set():
                return
            if (now - tts_last_audio_ts[0]) < ECHO_GUARD_AFTER_TTS_SEC:
                return
            try:
                raw = rec.recognize_google(audio, language=VOICE_LANGUAGE).strip()
            except Exception:
                return
            if not raw:
                return

            normalized = normalize_stt_text(raw)
            low = normalized.lower()
            talking_now = tts_playing_event.is_set() or model_generating_event.is_set()
            is_stop_music = any(x in low for x in ["para musica", "deten musica", "stop musica", "/musicoff", "/ytstop"])
            has_wake_prefix, after_wake = split_wake_prefix(normalized)

            if talking_now and (not ENABLE_INTERRUPTION):
                return

            # Evita falsos positivos cuando hay audio de salida del propio asistente:
            # en ese contexto se requiere palabra clave AL INICIO, salvo comando de parar musica.
            if talking_now and (not has_wake_prefix) and (not is_stop_music):
                return

            if has_wake_prefix:
                normalized = after_wake
                low = normalized.lower()
            if not normalized:
                return
            if is_probable_echo(normalized, last_spoken_reply[0], last_spoken_reply_ts[0]):
                return

            print(f"[STT] Transcrito: {raw}")
            if normalized != raw:
                print(f"[STT] Normalizado: {normalized}")

            if talking_now and ENABLE_INTERRUPTION:
                interrupt_event.set()
                flush_queue(tts_q)
                print("[INT] Interrupcion por voz detectada.")

            voice_q.put(normalized)

        stop_listening = recognizer.listen_in_background(
            microphone,
            _bg_callback,
            phrase_time_limit=MIC_PHRASE_TIME_LIMIT
        )

    print("\nChat iniciado")
    print(f"Personaje: {name} ({age} anos)")
    if sr is not None and recognizer is not None and microphone is not None:
        print("Entrada por voz activa (siempre).")
        if ENABLE_INTERRUPTION:
            print(f"Puedes interrumpir diciendo '{WAKE_WORD} ...' al inicio mientras habla.")
        else:
            print("Interrupcion por voz desactivada.")
        print("Puedes hablar aunque haya musica de YouTube.")
        print("Comandos por voz o teclado: /exit  /reset  /model <nombre>  /options <json>  /trust  /trustdebug  /yt <cancion>\n")
    else:
        print("Comandos: /exit  /reset  /model <nombre>  /options <json>  /trust  /trustdebug\n")

    try:
        while True:
            if sr is not None and recognizer is not None and microphone is not None:
                try:
                    user_input = voice_q.get(timeout=0.1).strip()
                except queue.Empty:
                    continue
            else:
                user_input = input("Tu: ").strip()
            if not user_input:
                continue

            if handle_youtube_intent(user_input, music_playing_event):
                continue

            if user_input == "/exit":
                print("Hasta luego!")
                break

            if user_input == "/reset":
                # VacÃƒÂ­a cola (para cortar habla pendiente)
                flush_queue(tts_q)
                flush_queue(voice_q)
                interrupt_event.clear()

                name, age = ask_identity()
                system_prompt = build_system_prompt(name, age)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": OUTPUT_STYLE_SYSTEM_PROMPT},
                ]
                print("Personaje reiniciado\n")
                continue
            if user_input.startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    model = parts[1].strip()
                    print(f"OK modelo cambiado a {model}")
                else:
                    print("Uso: /model llama3.2:3b")
                continue
            if user_input.startswith("/options"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print(f"Opciones actuales: {model_options}")
                else:
                    try:
                        new_opts = json.loads(parts[1])
                        if not isinstance(new_opts, dict):
                            raise ValueError("Las options deben ser un objeto JSON.")
                        model_options.update(new_opts)
                        print(f"OK options actualizadas: {model_options}")
                    except Exception as e:
                        print(f"[WARN] Options invalidas: {e}")
                continue
            if user_input == "/trust":
                print(f"Confianza actual: {trust}/100")
                print(trust_style_block(name, age, trust))
                continue
            if user_input == "/trustdebug":
                trust_debug = not trust_debug
                print(f"Trust debug: {'ON' if trust_debug else 'OFF'}")
                continue

            if ENABLE_INTERRUPTION and (tts_playing_event.is_set() or model_generating_event.is_set()):
                interrupt_event.set()
                flush_queue(tts_q)
                time.sleep(0.03)
            if ENABLE_INTERRUPTION:
                interrupt_event.clear()

            messages.append({"role": "user", "content": user_input})

            print(f"{name}:", end=" ", flush=True)

            reply = stream_chat_with_tts(
                model,
                messages,
                tts_q,
                model_options,
                LLM_KEEP_ALIVE,
                interrupt_event,
                model_generating_event,
            )

            if reply:
                messages.append({"role": "assistant", "content": reply})
                last_spoken_reply[0] = reply
                last_spoken_reply_ts[0] = time.time()
                # Modo rapido: evita segunda inferencia LLM para trust por defecto.
                if USE_TRUST_LLM:
                    delta, raw_trust = trust_delta_llm(user_input, reply, name)
                else:
                    next_trust = update_trust(user_input, trust)
                    delta = next_trust - trust
                    raw_trust = "heuristic"
                if trust_debug:
                    print(f"Trust raw: {raw_trust!r} -> delta {delta}")
                trust = max(0, min(100, trust + delta))
                state_msg = {"role": "system", "content": trust_style_block(name, age, trust)}
                if len(messages) > 1 and messages[1].get("role") == "system" and "Confianza hacia" in messages[1].get("content", ""):
                    messages[1] = state_msg
                else:
                    messages.insert(1, state_msg)
            else:
                messages.pop()
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")

    finally:
        # Cerrar worker TTS con limpieza
        stop_event.set()
        if stop_listening is not None:
            try:
                stop_listening(wait_for_stop=False)
            except Exception:
                pass
        tts_q.put(None)
        try:
            worker.join(timeout=1.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
