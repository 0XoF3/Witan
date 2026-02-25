import os
import asyncio
import logging
import io
import re
import sqlite3
import time
import base64
import html
from collections import defaultdict, deque
from datetime import datetime, timedelta
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
import discord
from discord import app_commands
import httpx
from openai import OpenAI
from better_profanity import profanity

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
NVIDIA_TOKEN = os.getenv("NVIDIA_TOKEN")
GOALS_DB_PATH = os.getenv("GOALS_DB_PATH", "witan_goals.db")
GOALS_DB_TIMEOUT_SECONDS = float(os.getenv("GOALS_DB_TIMEOUT_SECONDS", "3"))
GOAL_CONFIRMATION_TTL_SECONDS = int(os.getenv("GOAL_CONFIRMATION_TTL_SECONDS", "1200"))
WELCOME_DM_ON_MEMBER_JOIN = os.getenv("WELCOME_DM_ON_MEMBER_JOIN", "1").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_MEMBER_EVENTS = os.getenv("ENABLE_MEMBER_EVENTS", "1").strip().lower() in {"1", "true", "yes", "on"}
URL_SCAN_HTTP_TIMEOUT_SECONDS = float(os.getenv("URL_SCAN_HTTP_TIMEOUT_SECONDS", "10"))
URL_SCAN_WARN_THRESHOLD = int(os.getenv("URL_SCAN_WARN_THRESHOLD", "30"))
URL_SCAN_UNSAFE_THRESHOLD = int(os.getenv("URL_SCAN_UNSAFE_THRESHOLD", "70"))
URL_SCAN_CACHE_TTL_SECONDS = int(os.getenv("URL_SCAN_CACHE_TTL_SECONDS", "21600"))
URL_SCAN_ALLOWLIST_DOMAINS = {
    part.strip().lower()
    for part in (os.getenv("URL_SCAN_ALLOWLIST_DOMAINS") or "").split(",")
    if part.strip()
}
GOOGLE_SAFE_BROWSING_API_KEY = (
    os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
    or ""
).strip()
VIRUSTOTAL_API_KEY = (
    os.getenv("VIRUSTOTAL_API_KEY")
    or ""
).strip()
ALIENVAULT_OTX_API_KEY = (
    os.getenv("ALIENVAULT_OTX_API_KEY")
    or ""
).strip()

URL_SCAN_WARN_THRESHOLD = max(0, min(100, URL_SCAN_WARN_THRESHOLD))
URL_SCAN_UNSAFE_THRESHOLD = max(1, min(100, URL_SCAN_UNSAFE_THRESHOLD))
if URL_SCAN_WARN_THRESHOLD >= URL_SCAN_UNSAFE_THRESHOLD:
    URL_SCAN_WARN_THRESHOLD = max(0, URL_SCAN_UNSAFE_THRESHOLD - 1)

if not DISCORD_TOKEN or not NVIDIA_TOKEN:
    raise RuntimeError("DISCORD_TOKEN and NVIDIA_TOKEN must be set in .env")

# NVIDIA NIM client
client_ai = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_TOKEN,
)

# ---- EXACT MATCH WITH PLAYGROUND ----
NVIDIA_MODEL = "mistralai/mistral-7b-instruct-v0.3"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
NVIDIA_TEMPERATURE = float(os.getenv("NVIDIA_TEMPERATURE", "0.2"))
NVIDIA_TOP_P = float(os.getenv("NVIDIA_TOP_P", "0.7"))
NVIDIA_MAX_TOKENS = int(os.getenv("NVIDIA_MAX_TOKENS", "1024"))
NVIDIA_REQUEST_TIMEOUT = float(os.getenv("NVIDIA_REQUEST_TIMEOUT", "30"))
PROMPT_REPETITION_ENABLED = os.getenv("PROMPT_REPETITION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
PROMPT_REPETITION_TIMES = int(os.getenv("PROMPT_REPETITION_TIMES", "2"))
PROMPT_REPETITION_MAX_CHARS = int(os.getenv("PROMPT_REPETITION_MAX_CHARS", "4000"))

# ---------------- DISCORD SETUP ----------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = ENABLE_MEMBER_EVENTS
client = discord.Client(intents=intents)
COMMAND_TREE = app_commands.CommandTree(client)
APP_COMMANDS_SYNCED = False
NO_MENTIONS = discord.AllowedMentions.none()

# ---------------- STATE ----------------
MAX_PUBLIC_REPLIES = int(os.getenv("MAX_PUBLIC_REPLIES", "3"))

# Conversation memory (store last N messages, not turns)
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "1200"))
CONVERSATION_MEMORY = defaultdict(lambda: deque(maxlen=MEMORY_MAX_MESSAGES))

# Rate limiting and queueing
USER_COOLDOWN_SECONDS = float(os.getenv("USER_COOLDOWN_SECONDS", "3"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "6"))
MAX_USER_QUEUE = int(os.getenv("MAX_USER_QUEUE", "3"))
GLOBAL_AI_CONCURRENCY = int(os.getenv("GLOBAL_AI_CONCURRENCY", "3"))

USER_LAST_REQUEST_AT = {}
USER_REQUEST_TIMESTAMPS = defaultdict(deque)
USER_PENDING_COUNTS = defaultdict(int)
USER_LOCKS = {}
AI_SEMAPHORE = asyncio.Semaphore(GLOBAL_AI_CONCURRENCY)
INTRODUCED_USERS = set()
FEEDBACK_SESSIONS = {}
REDIRECT_SESSIONS = {}
REDIRECT_SESSION_TTL_SECONDS = int(os.getenv("REDIRECT_SESSION_TTL_SECONDS", "900"))
GOALS_DB_LOCK = asyncio.Lock()
PENDING_GOAL_CONFIRMATIONS = {}

BLOCKED_CATEGORY_COUNTS = defaultdict(int)

USER_REMINDER_TASK = None
USER_REMINDER_SCAN_INTERVAL_SECONDS = int(os.getenv("USER_REMINDER_SCAN_INTERVAL_SECONDS", "5"))
GOAL_REMINDER_TASK = None
GOAL_REMINDER_INTERVAL_SECONDS = int(os.getenv("GOAL_REMINDER_INTERVAL_SECONDS", "600"))
GOAL_REMINDER_WINDOW_SECONDS = int(os.getenv("GOAL_REMINDER_WINDOW_SECONDS", "18000"))
GOAL_REMINDER_REPLY_TTL_SECONDS = int(os.getenv("GOAL_REMINDER_REPLY_TTL_SECONDS", "86400"))
LEARNER_PROFILES = {}
USER_REMINDER_SETUP = {}
USER_REMINDERS = []
USER_REMINDER_NEXT_ID = 1
SENT_DAY_REMINDERS = set()
PENDING_DAY_CHECKS = {}
BOT_LOOP = None
URL_SCAN_CACHE = {}
TOTAL_URL_SCAN_PROVIDERS = 3
NEWS_API_KEY = (os.getenv("News_API") or os.getenv("NEWS_API_KEY") or "").strip()
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
NEWS_DEFAULT_TOPIC = os.getenv("NEWS_DEFAULT_TOPIC", "cybersecurity OR CVE OR malware").strip()
NEWS_ALLOWED_DOMAINS = {
    part.strip().lower()
    for part in (
        os.getenv(
            "NEWS_ALLOWED_DOMAINS",
            (
                "thehackernews.com,bleepingcomputer.com,securityweek.com,krebsonsecurity.com,"
                "darkreading.com,helpnetsecurity.com,infosecurity-magazine.com,securitymagazine.com,"
                "securityonline.info,cyberscoop.com,csoonline.com,scmagazine.com,hackread.com"
            ),
        )
        or ""
    ).split(",")
    if part.strip()
}
NEWS_DIGEST_SCAN_INTERVAL_SECONDS = int(os.getenv("NEWS_DIGEST_SCAN_INTERVAL_SECONDS", "30"))
NEWS_DIGEST_MAX_ITEMS = max(1, min(10, int(os.getenv("NEWS_DIGEST_MAX_ITEMS", "4"))))
NEWS_DIGEST_TASK = None

# ---------------- PROFANITY ----------------
profanity.load_censor_words()
profanity.add_censor_words(["kys", "kill yourself"])

# ---------------- SAFETY KEYWORDS ----------------
SAFETY_BLOCK_KEYWORDS = {
    "reverse shell", "bind shell", "revshell", "shellcode", "payload", "msfvenom", "meterpreter",
    "exploit code", "make malware", "ransomware", "keylogger", "backdoor", "bypass uac",
    "disable defender", "amsi bypass", "phishing kit"
}
SAFETY_BLOCK_PATTERNS = [
    (
        keyword,
        re.compile(
            rf"(?<![a-z0-9]){re.escape(keyword).replace(r'\\ ', r'\\s+')}(?![a-z0-9])",
            re.IGNORECASE,
        ),
    )
    for keyword in SAFETY_BLOCK_KEYWORDS
]

SAFETY_INTENT_RULES = {
    "malware": [
        r"\b(make|build|create|write|code|develop|generate)\b.{0,40}\b(malware|ransomware|trojan|virus|keylogger|backdoor|worm)\b",
        r"\b(how\s+to|help\s+me)\b.{0,40}\b(malware|ransomware|keylogger|backdoor)\b",
    ],
    "intrusion": [
        r"\b(exploit|hack|breach|pwn|bypass|privilege\s+escalation|uac\s+bypass)\b.{0,40}\b(code|script|steps|tool|payload)\b",
        r"\b(reverse\s+shell|bind\s+shell|shellcode|meterpreter|msfvenom)\b",
    ],
    "phishing": [
        r"\b(make|write|generate|create)\b.{0,40}\b(phishing|spoof|credential\s+harvest|fake\s+login)\b",
        r"\b(phishing\s+kit|email\s+template\s+to\s+steal\s+passwords?)\b",
    ],
    "self_harm": [
        r"\b(kill\s+myself|suicide|end\s+my\s+life|self\s*harm)\b",
    ],
}
SAFETY_INTENT_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    for category, patterns in SAFETY_INTENT_RULES.items()
}

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hii+|hello|hey|yo|what'?s\s*up|whatsup|sup)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
GREETING_REPLY = (
    "Hello, I'm Witan, your cybersecurity mentor and learning strategist.\n\n"
    "I can help you set clear learning goals, design structured study plans (including focused one-week roadmaps), "
    "recommend high-quality free resources, and guide you through real-world attack and defense concepts.\n\n"
    "I can also assist with certification preparation, practical lab guidance, skill assessments, and building a "
    "strong security mindset.\n\n"
    "How would you like to move forward?"
)
FIRST_TIME_INTRO = (
    "Hello, I'm Witan.\n"
    "Your cybersecurity mentor and learning strategist and much more.\n\n"
    "I help you:\n\n"
    "• Set clear, outcome-driven learning goals\n"
    "• Build structured study plans (including focused one-week roadmaps)\n"
    "• Master real-world attack and defense concepts\n"
    "• Prepare for certifications (OSCP, Security+, CEH, etc.)\n"
    "• Get practical lab guidance and skill assessments\n"
    "• Develop a disciplined security mindset\n\n"
    "Whether you're a beginner or leveling up, I'll give you structured direction not random advice.\n\n"
    "To get started, tell me:\n\n"
    "Your current level (Beginner / Intermediate / Advanced)\n\n"
    "Your goal (e.g., OSCP, bug bounty, SOC role, etc.)\n\n"
    "Let's build your roadmap.\n\n"
    "Or you want to ask some thing else like 'what is xss?' feel free to ask."
)
FEEDBACK_PROMPT = "Was this explanation clear? Reply `yes` or `no`."
EXPLOIT_REDIRECT_REPLY = (
    "I can't give working exploit instructions, but I can teach you this:\n"
    "1. Here's how defenders detect it\n"
    "2. Here's how to harden against it\n"
    "3. Here's how incident response works\n"
    "If you want, reply with 1, 2, or 3 for a detailed explanation. Or ask your next question normally."
)
URL_PATTERN = re.compile(r"https?://[^\s<>\]\"'`]+", re.IGNORECASE)
# ---------------- HELPERS ----------------
def sanitize_outgoing_text(text: str) -> str:
    if text is None:
        return ""

    # Prevent mention abuse from user-controlled content.
    safe = text.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
    safe = re.sub(r"<@!?(\d+)>", lambda m: f"<@\u200b{m.group(1)}>", safe)
    safe = re.sub(r"<@&(\d+)>", lambda m: f"<@&\u200b{m.group(1)}>", safe)
    safe = re.sub(r"<#(\d+)>", lambda m: f"<#\u200b{m.group(1)}>", safe)
    return safe


def contains_safety_keyword(text: str):
    normalized = re.sub(r"[_\-]+", " ", text.lower())
    for keyword, pattern in SAFETY_BLOCK_PATTERNS:
        if pattern.search(normalized):
            return keyword
    return None


def detect_safety_intent(text: str):
    normalized = " ".join(text.strip().split())
    for category, patterns in SAFETY_INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized):
                return category
    return None


def safe_alternative_for(category: str):
    if category in {"malware", "intrusion"}:
        return EXPLOIT_REDIRECT_REPLY
    if category == "phishing":
        return "I can help with anti-phishing training, email authentication (SPF/DKIM/DMARC), and detection guidance."
    if category == "self_harm":
        return "If you are in immediate danger, call emergency services. In the U.S., call or text 988 for immediate support."
    return "I can still help with safe, legal, and defensive guidance."


def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERN.match(text or ""))


def is_profane(text: str):
    return profanity.contains_profanity(text)


def extract_learner_level(text: str):
    t = (text or "").lower()
    if re.search(r"\b(beginner|newbie|new to|just starting)\b", t):
        return "Beginner"
    if re.search(r"\b(intermediate|mid[- ]?level)\b", t):
        return "Intermediate"
    if re.search(r"\b(advanced|expert|senior)\b", t):
        return "Advanced"
    return None


def extract_learner_goal(text: str):
    t = (text or "").strip()
    m = re.search(r"\b(?:goal|target|aim)\s*[:\-]?\s*([^\n.,;]{3,80})", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    goal_patterns = [
        (r"\boscp\b", "OSCP"),
        (r"\bsecurity\+\b", "Security+"),
        (r"\bceh\b", "CEH"),
        (r"\bbug\s*bounty\b", "Bug Bounty"),
        (r"\bsoc\b", "SOC Role"),
        (r"\bpenetration\s*testing\b|\bpentest\b", "Penetration Testing"),
        (r"\bred\s*team\b", "Red Team"),
        (r"\bblue\s*team\b", "Blue Team"),
    ]
    lowered = t.lower()
    for pattern, label in goal_patterns:
        if re.search(pattern, lowered):
            return label
    return None


def extract_learner_timeline(text: str):
    t = (text or "").strip()
    m = re.search(r"\btimeline\s*[:\-]?\s*([^\n.,;]{2,60})", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b(?:in|within|by)\s+(\d+\s*(?:day|days|week|weeks|month|months|year|years))\b", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b(\d+\s*(?:day|days|week|weeks|month|months|year|years))\b", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def update_learner_profile(user_id: int, text: str):
    profile = LEARNER_PROFILES.get(user_id, {}).copy()
    level = extract_learner_level(text)
    goal = extract_learner_goal(text)
    timeline = extract_learner_timeline(text)

    if level:
        profile["level"] = level
    if goal:
        profile["goal"] = goal
    if timeline:
        profile["timeline"] = timeline

    if profile:
        LEARNER_PROFILES[user_id] = profile


def get_learner_profile_context(user_id: int):
    profile = LEARNER_PROFILES.get(user_id) or {}
    if not profile:
        return None

    parts = []
    if profile.get("level"):
        parts.append(f"level={profile['level']}")
    if profile.get("goal"):
        parts.append(f"goal={profile['goal']}")
    if profile.get("timeline"):
        parts.append(f"timeline={profile['timeline']}")
    if not parts:
        return None
    return "User profile context: " + ", ".join(parts) + ". Personalize guidance accordingly."


def should_offer_feedback(user_text: str) -> bool:
    t = (user_text or "").lower()
    cues = [
        "simplify",
        "simpler",
        "easy way",
        "explain easy",
        "explain in easy",
        "explain simply",
        "beginner",
        "break it down",
    ]
    return any(cue in t for cue in cues)


def should_offer_goal_confirmation(user_text: str) -> bool:
    t = (user_text or "").lower()
    if re.search(r"\b(one|1)\s+week\b.*\bplan\b", t):
        return True
    if re.search(r"\bplan\b.*\b(one|1)\s+week\b", t):
        return True
    if re.search(r"\b(7|seven)\s+day\b.*\bplan\b", t):
        return True
    if re.search(r"\bplan\b.*\b(7|seven)\s+day\b", t):
        return True
    return False


def parse_reminder_datetime_input(text: str):
    raw = (text or "").strip()
    if not raw:
        return None

    now = datetime.now()
    raw_norm = re.sub(r"\s+", " ", raw).strip()
    candidate_formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %I:%M %p",
    ]
    for fmt in candidate_formats:
        try:
            dt = datetime.strptime(raw_norm, fmt)
            return dt
        except ValueError:
            continue

    # Natural language: today/tomorrow at time.
    m = re.match(r"^(today|tomorrow)\s+at\s+(\d{1,2}):(\d{2})(?:\s*([ap]m))?$", raw_norm, re.IGNORECASE)
    if m:
        day_word = m.group(1).lower()
        hour = int(m.group(2))
        minute = int(m.group(3))
        ampm = (m.group(4) or "").lower()

        if ampm:
            if hour < 1 or hour > 12:
                return None
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
        elif hour > 23:
            return None

        base_day = now.date() + timedelta(days=1 if day_word == "tomorrow" else 0)
        return datetime(base_day.year, base_day.month, base_day.day, hour, minute)

    # Natural language: month day at time (e.g., March 3 at 6:30 PM).
    month_day_formats = [
        "%B %d at %I:%M %p",
        "%b %d at %I:%M %p",
        "%B %d %I:%M %p",
        "%b %d %I:%M %p",
    ]
    for fmt in month_day_formats:
        with_year_fmt = f"{fmt} %Y"
        try:
            parsed = datetime.strptime(f"{raw_norm} {now.year}", with_year_fmt)
            candidate = parsed
            if candidate <= now:
                candidate = datetime.strptime(f"{raw_norm} {now.year + 1}", with_year_fmt)
            return candidate
        except ValueError:
            continue

    rel = re.search(r"\bin\s+(\d+)\s*(minute|minutes|hour|hours|day|days)\b", raw_norm, re.IGNORECASE)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        if "minute" in unit:
            return now + timedelta(minutes=amount)
        if "hour" in unit:
            return now + timedelta(hours=amount)
        if "day" in unit:
            return now + timedelta(days=amount)

    return None


def format_reminder_when_text(ts: int) -> str:
    now = datetime.now()
    dt = datetime.fromtimestamp(int(ts))
    today = now.date()
    tomorrow = today + timedelta(days=1)
    time_part = dt.strftime("%I:%M %p").lstrip("0")
    if dt.date() == today:
        return f"today at {time_part}"
    if dt.date() == tomorrow:
        return f"tomorrow at {time_part}"
    return f"{dt.strftime('%B %d, %Y')} at {time_part}"


def parse_daily_time_input(text: str):
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return None
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.hour, parsed.minute
        except ValueError:
            continue
    return None


def _clean_html_text(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _topic_terms(topic_query: str):
    text = (topic_query or "").lower()
    text = re.sub(r"[()\"']", " ", text)
    parts = re.split(r"\s+|,|\bor\b|\band\b", text)
    terms = []
    for part in parts:
        t = part.strip()
        if len(t) < 3:
            continue
        if t in {"the", "and", "for", "with", "news", "daily", "latest", "bug", "bounty"}:
            continue
        if t not in terms:
            terms.append(t)
    return terms[:10]


def _looks_english_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    latin_letters = len(re.findall(r"[A-Za-z]", sample))
    cjk_chars = len(re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", sample))
    if cjk_chars > 6 and latin_letters < 6:
        return False
    return latin_letters >= 6


def _is_cyber_relevant_article(title: str, snippet: str, topic_query: str) -> bool:
    hay = f"{title or ''} {snippet or ''}".lower()
    non_cyber_block_terms = [
        "movie", "film", "novel", "box office", "celebrity", "trailer",
        "murder", "kill", "crime drama", "music video",
    ]
    if any(term in hay for term in non_cyber_block_terms):
        # Allow through only if there are very strong cyber anchors.
        if not re.search(r"\bcve-\d{4}-\d{3,7}\b", hay) and "vulnerability" not in hay and "exploit" not in hay:
            return False
    if re.search(r"\bcve-\d{4}-\d{3,7}\b", hay):
        return True
    cyber_keywords = [
        "cybersecurity", "cyber security", "infosec", "malware", "ransomware",
        "phishing", "vulnerability", "exploit", "zero-day", "zeroday",
        "data breach", "threat", "apt", "botnet", "soc", "siem", "edr",
        "xss", "sqli", "sql injection", "ddos", "firewall", "incident response",
        "bug bounty", "bugbounty",
    ]
    cyber_hits = sum(1 for k in cyber_keywords if k in hay)
    if cyber_hits >= 1:
        return True
    topic_phrase = re.sub(r"\s+", " ", (topic_query or "").strip().lower())
    if topic_phrase and len(topic_phrase) >= 5 and topic_phrase in hay and cyber_hits >= 1:
        return True
    topic_terms = _topic_terms(topic_query)
    topic_hits = sum(1 for term in topic_terms if term in hay)
    if topic_hits >= 2 and cyber_hits >= 1:
        return True
    return False


def _is_allowed_news_domain(url: str) -> bool:
    if not NEWS_ALLOWED_DOMAINS:
        return True
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    if host in NEWS_ALLOWED_DOMAINS:
        return True
    return any(host.endswith(f".{domain}") for domain in NEWS_ALLOWED_DOMAINS)


def _format_news_published_date(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if not text:
        return "Unknown date"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return text[:10]


async def fetch_cyber_news_items(topic_query: str, limit: int = 4):
    query = (topic_query or "").strip() or NEWS_DEFAULT_TOPIC
    if not NEWS_API_KEY:
        logging.error("NewsAPI key is missing; cannot fetch news digest items.")
        return []
    normalized = re.sub(r"\s*,\s*", " OR ", query)
    strong_cyber_query = (
        '(cve OR vulnerability OR exploit OR "zero-day" OR malware OR ransomware OR phishing OR "data breach") '
        'AND ("cybersecurity" OR "security flaw" OR "threat intel") '
        'NOT (movie OR film OR celebrity OR sports OR music OR politics)'
    )
    queries = []
    for candidate in [
        normalized,
        f"({normalized}) AND (cybersecurity OR cve OR malware OR ransomware OR vulnerability)",
        NEWS_DEFAULT_TOPIC,
        strong_cyber_query,
    ]:
        text = (candidate or "").strip()
        if text and text not in queries:
            queries.append(text)

    timeout = httpx.Timeout(15)
    for restrict_domains in [True, False]:
        for q in queries:
            params = {
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max(10, min(50, int(limit) * 10)),
                "apiKey": NEWS_API_KEY,
            }
            if restrict_domains and NEWS_ALLOWED_DOMAINS:
                params["domains"] = ",".join(sorted(NEWS_ALLOWED_DOMAINS))
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http_client:
                    response = await http_client.get(NEWS_API_ENDPOINT, params=params)
                    response.raise_for_status()
                    body = response.json() or {}
                if str(body.get("status", "")).lower() != "ok":
                    logging.warning("NewsAPI returned non-ok status for query=%s", q)
                    continue
                articles = body.get("articles") or []
                if not articles:
                    continue
                items = []
                for article in articles:
                    title = (article.get("title") or "Untitled").strip()
                    article_url = (article.get("url") or "").strip() or "https://newsapi.org/"
                    if restrict_domains and not _is_allowed_news_domain(article_url):
                        continue
                    source_text = _clean_html_text(
                        (article.get("description") or "")
                        + " "
                        + (article.get("content") or "")
                    )
                    if not _looks_english_text(title + " " + source_text):
                        continue
                    if not _is_cyber_relevant_article(title, source_text, q):
                        continue
                    items.append(
                        {
                            "title": title,
                            "published": _format_news_published_date(article.get("publishedAt") or ""),
                            "url": article_url,
                            "source_text": source_text,
                        }
                    )
                    if len(items) >= max(1, limit):
                        break
                if items:
                    return items
            except Exception:
                logging.exception("NewsAPI fetch failed for query=%s", q)
                continue
    return []


async def summarize_news_item(title: str, source_text: str):
    fallback = (source_text or "").strip()
    if len(fallback) > 240:
        fallback = fallback[:237].rstrip() + "..."
    if not fallback:
        fallback = "New cybersecurity update published. Open the article for full details."

    prompt = (
        "Summarize this cybersecurity news article in 1-2 plain sentences "
        "(max 220 characters), defensive and factual tone.\n"
        f"Title: {title}\n"
        f"Snippet: {(source_text or '')[:1500]}"
    )
    messages = [
        {"role": "system", "content": "You produce concise cybersecurity news summaries."},
        {"role": "user", "content": prompt},
    ]
    reply = await query_ai(messages)
    if not reply:
        return fallback
    cleaned = re.sub(r"\s+", " ", str(reply)).strip().strip("\"")
    if not cleaned:
        return fallback
    if len(cleaned) > 260:
        cleaned = cleaned[:257].rstrip() + "..."
    return cleaned


async def build_daily_news_embed(topic_query: str):
    query = (topic_query or "").strip() or NEWS_DEFAULT_TOPIC
    items = await fetch_cyber_news_items(query, limit=NEWS_DIGEST_MAX_ITEMS)
    if not items:
        return None

    embed = discord.Embed(
        title="🔐 Weekly Cybersecurity News",
        color=discord.Color.yellow(),
    )
    embed.add_field(name="🎯 Topic", value=query[:1024], inline=False)
    embed.add_field(name="\u200b", value="\u200b", inline=False)

    for idx, item in enumerate(items):
        summary = await summarize_news_item(item["title"], item["source_text"])
        field_name = f"📰 {item['title'][:240]}"
        field_value = (
            f"📅 {item['published']}\n\n"
            f"🔗 [Read More]({item['url']})\n\n"
            f"📝 {summary}"
        )
        if len(field_value) > 1024:
            field_value = field_value[:1021].rstrip() + "..."
        embed.add_field(name=field_name, value=field_value, inline=False)
        if idx < len(items) - 1:
            embed.add_field(name="\u200b", value="━━━━━━━━━━━━━━━━━━━━━━", inline=False)
    return embed


def get_datetime_reply(text: str):
    t = (text or "").strip().lower()
    if not t:
        return None

    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p").lstrip("0")
    tomorrow = now + timedelta(days=1)
    yesterday = now - timedelta(days=1)
    day_after_tomorrow = now + timedelta(days=2)
    day_before_yesterday = now - timedelta(days=2)

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%A, %B %d, %Y")

    # Relative date questions
    if re.search(r"\b(day\s+after\s+tomorrow)\b", t):
        return f"The date for day after tomorrow is {_fmt(day_after_tomorrow)}."
    if re.search(r"\b(day\s+before\s+yesterday)\b", t):
        return f"The date for day before yesterday is {_fmt(day_before_yesterday)}."
    if re.search(r"\b(tomorrow('?s)?\s+date|date\s+tomorrow|what\s+is\s+tomorrow('?s)?\s+date|what\s+date\s+is\s+tomorrow)\b", t):
        return f"The date for tomorrow is {_fmt(tomorrow)}."
    if re.search(r"\b(yesterday('?s)?\s+date|date\s+yesterday|what\s+is\s+yesterday('?s)?\s+date|what\s+date\s+was\s+yesterday)\b", t):
        return f"The date for yesterday was {_fmt(yesterday)}."

    date_patterns = [
        r"\b(today'?s?\s+date|date\s+today|what\s+is\s+the\s+date|current\s+date)\b",
        r"\bwhat\s+day\s+is\s+it\b",
        r"\bday\s+today\b",
    ]
    time_patterns = [
        r"\b(current\s+time|time\s+now|what\s+time\s+is\s+it|time\s+please)\b",
    ]
    datetime_patterns = [
        r"\b(date\s+and\s+time|time\s+and\s+date)\b",
    ]

    if any(re.search(pattern, t) for pattern in datetime_patterns):
        return f"Current date and time: {date_str}, {time_str}."
    if any(re.search(pattern, t) for pattern in date_patterns):
        return f"Today's date is {date_str}."
    if any(re.search(pattern, t) for pattern in time_patterns):
        return f"Current time is {time_str}."
    return None


def extract_urls_from_text(text: str):
    raw = URL_PATTERN.findall(text or "")
    if not raw:
        return []
    cleaned = []
    seen = set()
    for url in raw:
        normalized = url.rstrip(".,!?;:)]}")
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_url_allowlisted(url: str) -> bool:
    host = _url_host(url)
    if not host:
        return False
    if host in URL_SCAN_ALLOWLIST_DOMAINS:
        return True
    return any(host.endswith(f".{domain}") for domain in URL_SCAN_ALLOWLIST_DOMAINS)


def _cache_key_for_url(url: str) -> str:
    return url.strip().lower()


def _get_cached_url_scan(url: str):
    key = _cache_key_for_url(url)
    item = URL_SCAN_CACHE.get(key)
    if not item:
        return None
    expires_at = float(item.get("expires_at", 0))
    if expires_at < time.time():
        URL_SCAN_CACHE.pop(key, None)
        return None
    data = item.get("data")
    if not data:
        return None
    copied = dict(data)
    copied["cached"] = True
    return copied


def _set_cached_url_scan(url: str, data):
    key = _cache_key_for_url(url)
    URL_SCAN_CACHE[key] = {
        "expires_at": time.time() + max(60, URL_SCAN_CACHE_TTL_SECONDS),
        "data": data,
    }


def _virustotal_url_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


async def _scan_google_safe_browsing(http_client: httpx.AsyncClient, url: str):
    if not GOOGLE_SAFE_BROWSING_API_KEY:
        return {"source": "google_safe_browsing", "available": False, "flagged": False, "score": 0, "details": []}

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_SAFE_BROWSING_API_KEY}"
    payload = {
        "client": {"clientId": "witan-bot", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        res = await http_client.post(endpoint, json=payload)
        if res.status_code >= 400:
            return {"source": "google_safe_browsing", "available": False, "flagged": False, "score": 0, "details": []}
        matches = (res.json() or {}).get("matches") or []
        threat_types = sorted({m.get("threatType", "UNKNOWN") for m in matches})
        flagged = bool(matches)
        return {
            "source": "google_safe_browsing",
            "available": True,
            "flagged": flagged,
            "score": 45 if flagged else 0,
            "details": threat_types,
        }
    except Exception:
        logging.exception("Google Safe Browsing check failed for url=%s", url)
        return {"source": "google_safe_browsing", "available": False, "flagged": False, "score": 0, "details": []}


async def _scan_virustotal(http_client: httpx.AsyncClient, url: str):
    if not VIRUSTOTAL_API_KEY:
        return {"source": "virustotal", "available": False, "flagged": False, "score": 0, "details": []}

    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    url_id = _virustotal_url_id(url)
    endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
    try:
        res = await http_client.get(endpoint, headers=headers)
        if res.status_code == 404:
            submit = await http_client.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url})
            if submit.status_code >= 400:
                return {"source": "virustotal", "available": False, "flagged": False, "score": 0, "details": []}
            return {"source": "virustotal", "available": True, "flagged": False, "score": 0, "details": ["submitted_for_analysis"]}
        if res.status_code >= 400:
            return {"source": "virustotal", "available": False, "flagged": False, "score": 0, "details": []}

        attrs = ((res.json() or {}).get("data") or {}).get("attributes") or {}
        stats = attrs.get("last_analysis_stats") or {}
        malicious = int(stats.get("malicious", 0) or 0)
        suspicious = int(stats.get("suspicious", 0) or 0)
        harmless = int(stats.get("harmless", 0) or 0)
        flagged = malicious > 0 or suspicious > 0
        weighted = min(35, malicious * 8 + suspicious * 4)
        details = [f"malicious={malicious}", f"suspicious={suspicious}", f"harmless={harmless}"]
        return {
            "source": "virustotal",
            "available": True,
            "flagged": flagged,
            "score": weighted,
            "details": details,
        }
    except Exception:
        logging.exception("VirusTotal check failed for url=%s", url)
        return {"source": "virustotal", "available": False, "flagged": False, "score": 0, "details": []}


async def _scan_alienvault_otx(http_client: httpx.AsyncClient, url: str):
    headers = {}
    if ALIENVAULT_OTX_API_KEY:
        headers["X-OTX-API-KEY"] = ALIENVAULT_OTX_API_KEY
    endpoint = f"https://otx.alienvault.com/api/v1/indicators/url/{quote(url, safe='')}/general"
    try:
        res = await http_client.get(endpoint, headers=headers)
        if res.status_code >= 400:
            return {"source": "alienvault_otx", "available": False, "flagged": False, "score": 0, "details": []}
        body = res.json() or {}
        pulse_info = body.get("pulse_info") or {}
        pulse_count = int(pulse_info.get("count", 0) or 0)
        flagged = pulse_count > 0
        score = min(20, pulse_count * 4) if flagged else 0
        return {
            "source": "alienvault_otx",
            "available": True,
            "flagged": flagged,
            "score": score,
            "details": [f"pulses={pulse_count}"],
        }
    except Exception:
        logging.exception("AlienVault OTX check failed for url=%s", url)
        return {"source": "alienvault_otx", "available": False, "flagged": False, "score": 0, "details": []}


def aggregate_url_scan_results(provider_results):
    total_score = 0
    providers_used = 0
    flagged_sources = []
    unavailable_sources = []
    provider_summaries = []

    for result in provider_results:
        if result.get("available"):
            providers_used += 1
        else:
            unavailable_sources.append(result.get("source", "unknown"))
        source = result.get("source", "unknown")
        score = int(result.get("score", 0) or 0)
        total_score += score
        details = ", ".join(result.get("details") or [])
        provider_summaries.append(f"{source}: {'flagged' if result.get('flagged') else 'clear'}{(' (' + details + ')') if details else ''}")
        if result.get("flagged"):
            flagged_sources.append(source)

    total_score = min(100, total_score)
    if providers_used == 0:
        level = "not_sure"
    elif total_score == 0:
        level = "safe"
    elif total_score >= URL_SCAN_UNSAFE_THRESHOLD:
        level = "unsafe"
    elif total_score >= URL_SCAN_WARN_THRESHOLD:
        level = "suspicious"
    else:
        level = "likely_safe"

    return {
        "risk_score": total_score,
        "risk_level": level,
        "providers_used": providers_used,
        "providers_total": TOTAL_URL_SCAN_PROVIDERS,
        "unavailable_sources": unavailable_sources,
        "flagged_sources": flagged_sources,
        "provider_summaries": provider_summaries,
    }


async def scan_single_url(url: str):
    cached = _get_cached_url_scan(url)
    if cached:
        return cached

    timeout = httpx.Timeout(URL_SCAN_HTTP_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        results = await asyncio.gather(
            _scan_google_safe_browsing(http_client, url),
            _scan_virustotal(http_client, url),
            _scan_alienvault_otx(http_client, url),
        )
    aggregate = aggregate_url_scan_results(results)
    scan_data = {"url": url, "providers": results, "aggregate": aggregate, "cached": False}
    _set_cached_url_scan(url, scan_data)
    return scan_data


def _scan_level_color(level: str):
    if level == "unsafe":
        return discord.Color.red()
    if level == "suspicious":
        return discord.Color.orange()
    if level == "not_sure":
        return discord.Color.light_grey()
    return discord.Color.green()


def build_url_warning_embed(risky_entries):
    top_level = "unsafe" if any(item["aggregate"]["risk_level"] == "unsafe" for item in risky_entries) else "suspicious"
    title = "Link Safety Warning"
    description = (
        "**Potentially dangerous link detected.**\n"
        "Verification used: Google Safe Browsing, VirusTotal, AlienVault OTX."
    )
    embed = discord.Embed(title=title, description=description, color=_scan_level_color(top_level))
    for entry in risky_entries[:5]:
        agg = entry["aggregate"]
        sources = ", ".join(agg["flagged_sources"]) if agg["flagged_sources"] else "none"
        unavailable = ", ".join(agg["unavailable_sources"]) if agg["unavailable_sources"] else "none"
        level = str(agg["risk_level"]).upper()
        value = (
            f"**RISK:** {level} (**{agg['risk_score']}%**)\n"
            f"**FLAGGED BY:** {sources}\n"
            f"**PROVIDER COVERAGE:** {agg['providers_used']}/{agg['providers_total']} (unavailable: {unavailable})\n"
            f"**URL:** {entry['url']}"
        )
        embed.add_field(name="Detected URL", value=value[:1024], inline=False)
    omitted = len(risky_entries) - 5
    footer = "This is an automated warning. Review before clicking unknown links."
    if omitted > 0:
        footer = f"{omitted} more risky URL(s) were omitted."
    embed.set_footer(text=footer)
    return embed


def build_url_scan_result_embed(scan_result, requested_by: str = ""):
    agg = scan_result["aggregate"]
    level = agg["risk_level"]
    level_label = {
        "unsafe": "UNSAFE",
        "suspicious": "SUSPICIOUS",
        "not_sure": "NOT SURE",
        "safe": "SAFE",
        "likely_safe": "LIKELY SAFE",
    }.get(level, str(level).upper())

    provider_labels = {
        "google_safe_browsing": "Google Safe Browsing",
        "virustotal": "VirusTotal",
        "alienvault_otx": "AlienVault OTX",
    }

    status_icon = {
        "unsafe": "🔴",
        "suspicious": "🟠",
        "not_sure": "⚪",
        "safe": "🟢",
        "likely_safe": "🟢",
    }.get(level, "ℹ️")

    try:
        parsed = urlparse(scan_result["url"])
        target = (parsed.netloc + parsed.path + ((f"?{parsed.query}") if parsed.query else "")).strip() or scan_result["url"]
    except Exception:
        target = scan_result["url"]

    detection_lines = []
    for provider in scan_result.get("providers", []):
        source = provider.get("source", "unknown")
        name = provider_labels.get(source, source.replace("_", " ").title())
        available = bool(provider.get("available"))
        flagged = bool(provider.get("flagged"))
        details = ", ".join(provider.get("details") or [])

        if not available:
            line = f"⚪ {name} — Unavailable"
        elif flagged:
            line = f"❌ {name} — Threats Detected"
        else:
            if source == "google_safe_browsing":
                line = f"✔ {name} — Clean"
            elif source == "virustotal":
                line = f"✔ {name} — Clean"
            elif source == "alienvault_otx":
                line = f"✔ {name} — No Threat Intel"
            else:
                line = f"✔ {name} — Clean"
        if details and available and flagged:
            line += f" ({details})"
        detection_lines.append(line)

    unavailable_count = len(agg["unavailable_sources"])
    cache_note = "✅ Cached" if scan_result.get("cached") else "❌ Not Cached"
    requested_line = requested_by or "Unknown"
    summary_block = "\n".join(detection_lines) if detection_lines else "No provider data."

    description = (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{status_icon} Status: **{level_label}**\n"
        f"📊 Risk Score: **{agg['risk_score']}%**\n\n"
        "🔗 Target\n"
        f"{target}\n\n"
        "🧠 Detection Summary\n"
        f"{summary_block}\n\n"
        "📦 Provider Coverage\n"
        f"{agg['providers_used']} / {agg['providers_total']} Active • {unavailable_count} Unavailable\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Requested by: {requested_line}\n"
        f"Cache: {cache_note}"
    )
    embed = discord.Embed(
        title="🛡️ URL Security Report",
        description=description[:4096],
        color=_scan_level_color(level),
    )
    return embed


async def scan_message_links_and_warn(message: discord.Message, text: str):
    urls = extract_urls_from_text(text)
    if not urls:
        return
    urls_to_scan = [url for url in urls if not is_url_allowlisted(url)]
    if not urls_to_scan:
        return
    scan_results = await asyncio.gather(*(scan_single_url(url) for url in urls_to_scan))
    risky = [item for item in scan_results if item["aggregate"]["risk_level"] in {"suspicious", "unsafe"}]
    if not risky:
        return

    try:
        for item in risky[:3]:
            embed = build_url_scan_result_embed(item, requested_by=str(message.author))
            await message.channel.send(embed=embed, allowed_mentions=NO_MENTIONS)
        remaining = len(risky) - 3
        if remaining > 0:
            await message.channel.send(
                f"⚠️ {remaining} additional risky URL(s) were detected but not shown.",
                allowed_mentions=NO_MENTIONS,
            )
    except Exception:
        logging.exception("Failed to send URL scan warning for message id=%s", message.id)


def log_url_scan_provider_config():
    providers = {
        "google_safe_browsing": bool(GOOGLE_SAFE_BROWSING_API_KEY),
        "virustotal": bool(VIRUSTOTAL_API_KEY),
        "alienvault_otx": bool(ALIENVAULT_OTX_API_KEY),
    }
    configured = [name for name, ok in providers.items() if ok]
    missing = [name for name, ok in providers.items() if not ok]
    logging.info("URL scan providers configured: %s/%s (%s)", len(configured), TOTAL_URL_SCAN_PROVIDERS, ", ".join(configured) or "none")
    if missing:
        logging.warning("URL scan providers missing keys/config: %s", ", ".join(missing))
    logging.info(
        "URL scan thresholds warn=%s unsafe=%s cache_ttl=%ss allowlist=%s",
        URL_SCAN_WARN_THRESHOLD,
        URL_SCAN_UNSAFE_THRESHOLD,
        URL_SCAN_CACHE_TTL_SECONDS,
        ", ".join(sorted(URL_SCAN_ALLOWLIST_DOMAINS)) or "(none)",
    )


async def run_provider_self_test(sample_url: str = "https://example.com"):
    timeout = httpx.Timeout(URL_SCAN_HTTP_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        results = await asyncio.gather(
            _scan_google_safe_browsing(http_client, sample_url),
            _scan_virustotal(http_client, sample_url),
            _scan_alienvault_otx(http_client, sample_url),
        )
    for result in results:
        logging.info(
            "URL scan self-test provider=%s available=%s flagged=%s details=%s",
            result.get("source"),
            result.get("available"),
            result.get("flagged"),
            ", ".join(result.get("details") or []) or "(none)",
        )


def get_goal_confirmation_key(message: discord.Message):
    return (message.channel.id, message.author.id)


def _ensure_column(conn, table: str, column_name: str, column_definition: str):
    # Lightweight runtime migration so older DB files keep working after new releases.
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_definition}")


def init_goals_db():
    with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
        # WAL + NORMAL improves concurrent read/write behavior for this async bot workload.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                goal_text TEXT NOT NULL,
                plan_text TEXT,
                duration_days INTEGER NOT NULL DEFAULT 7,
                status TEXT NOT NULL DEFAULT 'active',
                completed_at INTEGER,
                created_at INTEGER NOT NULL
            )
            """
        )
        _ensure_column(conn, "goals", "plan_text", "plan_text TEXT")
        _ensure_column(conn, "goals", "duration_days", "duration_days INTEGER NOT NULL DEFAULT 7")
        _ensure_column(conn, "goals", "status", "status TEXT NOT NULL DEFAULT 'active'")
        _ensure_column(conn, "goals", "completed_at", "completed_at INTEGER")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goal_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                day_number INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(goal_id, day_number)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_subscriptions (
                user_id TEXT PRIMARY KEY,
                send_hour INTEGER NOT NULL,
                send_minute INTEGER NOT NULL,
                topic_query TEXT NOT NULL DEFAULT 'cybersecurity OR CVE OR malware',
                is_enabled INTEGER NOT NULL DEFAULT 1,
                last_sent_date TEXT,
                updated_at INTEGER NOT NULL
            )
            """
        )
        _ensure_column(
            conn,
            "news_subscriptions",
            "topic_query",
            "topic_query TEXT NOT NULL DEFAULT 'cybersecurity OR CVE OR malware'",
        )
        conn.execute(
            """
            UPDATE news_subscriptions
            SET topic_query = COALESCE(NULLIF(TRIM(topic_query), ''), ?)
            """,
            (NEWS_DEFAULT_TOPIC,),
        )
        conn.commit()


async def upsert_news_subscription(user_id: int, send_hour: int, send_minute: int, topic_query: str):
    now_ts = int(time.time())
    topic_value = (topic_query or "").strip()[:200]
    if not topic_value:
        topic_value = NEWS_DEFAULT_TOPIC

    def _upsert():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.execute(
                """
                INSERT INTO news_subscriptions (user_id, send_hour, send_minute, topic_query, is_enabled, last_sent_date, updated_at)
                VALUES (?, ?, ?, ?, 1, NULL, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    send_hour = excluded.send_hour,
                    send_minute = excluded.send_minute,
                    topic_query = excluded.topic_query,
                    is_enabled = 1,
                    last_sent_date = NULL,
                    updated_at = excluded.updated_at
                """,
                (str(user_id), int(send_hour), int(send_minute), topic_value, now_ts),
            )
            conn.commit()

    async with GOALS_DB_LOCK:
        await asyncio.to_thread(_upsert)


async def disable_news_subscription(user_id: int):
    now_ts = int(time.time())

    def _disable():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.execute(
                """
                UPDATE news_subscriptions
                SET is_enabled = 0, updated_at = ?
                WHERE user_id = ?
                """,
                (now_ts, str(user_id)),
            )
            conn.commit()

    async with GOALS_DB_LOCK:
        await asyncio.to_thread(_disable)


async def get_due_news_subscriptions(now_local: datetime):
    today = now_local.strftime("%Y-%m-%d")
    hour = int(now_local.hour)
    minute = int(now_local.minute)

    def _select():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT user_id, send_hour, send_minute, topic_query, last_sent_date
                FROM news_subscriptions
                WHERE is_enabled = 1
                  AND send_hour = ?
                  AND send_minute = ?
                  AND (last_sent_date IS NULL OR last_sent_date <> ?)
                """,
                (hour, minute, today),
            ).fetchall()

    async with GOALS_DB_LOCK:
        rows = await asyncio.to_thread(_select)
    return [dict(row) for row in rows]


async def mark_news_subscription_sent(user_id: int, sent_date: str):
    now_ts = int(time.time())

    def _mark():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.execute(
                """
                UPDATE news_subscriptions
                SET last_sent_date = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (sent_date, now_ts, str(user_id)),
            )
            conn.commit()

    async with GOALS_DB_LOCK:
        await asyncio.to_thread(_mark)


async def save_goal_to_db(message: discord.Message, goal_text: str, plan_text: str = None, duration_days: int = 7):
    goal_value = goal_text[:500]
    plan_value = (plan_text or "")[:4000] if plan_text else None
    duration_value = max(1, min(365, int(duration_days)))
    user_id = str(message.author.id)
    username = str(message.author)
    channel_id = str(message.channel.id)
    created_at = int(time.time())

    def _insert():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.execute(
                """
                INSERT INTO goals (user_id, username, channel_id, goal_text, plan_text, duration_days, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (user_id, username, channel_id, goal_value, plan_value, duration_value, created_at),
            )
            conn.commit()
            return cursor.lastrowid

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_insert)


async def get_active_goal_for_user(user_id: int):
    def _select():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT id, goal_text, plan_text, duration_days, created_at
                FROM goals
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_select)


async def get_active_goal_snapshot_for_user(user_id: int):
    def _select():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            goal = conn.execute(
                """
                SELECT id, goal_text, duration_days, created_at
                FROM goals
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
            if not goal:
                return None

            row = conn.execute(
                """
                SELECT COUNT(*) AS completed_count, COALESCE(MAX(day_number), 0) AS max_day
                FROM goal_progress
                WHERE goal_id = ?
                """,
                (int(goal["id"]),),
            ).fetchone()

            completed_count = int(row["completed_count"] or 0)
            max_day = int(row["max_day"] or 0)
            return {
                "goal_id": int(goal["id"]),
                "goal_text": str(goal["goal_text"]),
                "duration_days": int(goal["duration_days"] or 7),
                "created_at": int(goal["created_at"]),
                "completed_count": completed_count,
                "max_day": max_day,
                "progressed_days": max(completed_count, max_day),
            }

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_select)


async def reset_active_goal_for_user(user_id: int):
    now = int(time.time())

    def _reset():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            goal = conn.execute(
                """
                SELECT id, goal_text
                FROM goals
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
            if not goal:
                return {"status": "no_goal"}

            goal_id = int(goal["id"])
            conn.execute("DELETE FROM goal_progress WHERE goal_id = ?", (goal_id,))
            conn.execute(
                "UPDATE goals SET status = 'reset', completed_at = ? WHERE id = ?",
                (now, goal_id),
            )
            conn.commit()
            return {"status": "ok", "goal_text": str(goal["goal_text"]), "goal_id": goal_id}

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_reset)


async def get_due_goal_reminders(now_ts: int):
    def _select():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, user_id, goal_text, duration_days, created_at
                FROM goals
                WHERE status = 'active'
                ORDER BY created_at ASC
                """
            ).fetchall()

            due = []
            for row in rows:
                goal_id = int(row["id"])
                user_id = int(row["user_id"])
                duration_days = int(row["duration_days"] or 7)
                created_at = int(row["created_at"])
                elapsed = max(0, now_ts - created_at)
                day_index = (elapsed // 86400) + 1
                if day_index < 1 or day_index > duration_days:
                    continue

                day_elapsed = elapsed % 86400
                seconds_until_rollover = 86400 - day_elapsed
                if seconds_until_rollover > GOAL_REMINDER_WINDOW_SECONDS:
                    continue

                max_day = conn.execute(
                    """
                    SELECT COALESCE(MAX(day_number), 0) AS max_day
                    FROM goal_progress
                    WHERE goal_id = ?
                    """,
                    (goal_id,),
                ).fetchone()["max_day"]
                max_day = int(max_day or 0)
                if max_day >= day_index:
                    continue

                due.append(
                    {
                        "goal_id": goal_id,
                        "user_id": user_id,
                        "day_number": int(day_index),
                        "goal_text": str(row["goal_text"]),
                    }
                )
            return due

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_select)


async def record_goal_day_completion(user_id: int, day_number: int):
    now = int(time.time())

    def _record():
        with sqlite3.connect(GOALS_DB_PATH, timeout=GOALS_DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            goal = conn.execute(
                """
                SELECT id, duration_days, created_at
                FROM goals
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
            if not goal:
                return {"status": "no_goal"}

            goal_id = int(goal["id"])
            duration_days = int(goal["duration_days"] or 7)
            created_at = int(goal["created_at"])

            try:
                conn.execute(
                    """
                    INSERT INTO goal_progress (goal_id, user_id, day_number, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (goal_id, str(user_id), day_number, now),
                )
            except sqlite3.IntegrityError:
                return {"status": "already_done", "duration_days": duration_days}

            completed_count = conn.execute(
                "SELECT COUNT(*) FROM goal_progress WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()[0]

            completed_goal = False
            if day_number >= duration_days:
                conn.execute(
                    "UPDATE goals SET status = 'completed', completed_at = ? WHERE id = ?",
                    (now, goal_id),
                )
                completed_goal = True

            conn.commit()
            return {
                "status": "ok",
                "duration_days": duration_days,
                "created_at": created_at,
                "completed_count": int(completed_count),
                "completed_goal": completed_goal,
            }

    async with GOALS_DB_LOCK:
        return await asyncio.to_thread(_record)


def get_memory_key(message: discord.Message):
    return ("dm", message.channel.id, message.author.id)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_messages_with_memory(memory_key, user_text: str):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    user_id = memory_key[2] if isinstance(memory_key, tuple) and len(memory_key) >= 3 else None
    if user_id is not None:
        profile_context = get_learner_profile_context(int(user_id))
        if profile_context:
            messages.append({"role": "system", "content": profile_context})

    history = list(CONVERSATION_MEMORY[memory_key])
    token_budget = MEMORY_MAX_TOKENS
    selected_history = []

    for item in reversed(history):
        token_cost = estimate_tokens(item["content"])
        if token_cost > token_budget:
            break
        token_budget -= token_cost
        selected_history.append(item)

    selected_history.reverse()
    messages.extend(selected_history)
    messages.append({"role": "user", "content": user_text})
    return messages


def save_turn(memory_key, user_text: str, assistant_text: str):
    conversation = CONVERSATION_MEMORY[memory_key]
    conversation.append({"role": "user", "content": user_text})
    conversation.append({"role": "assistant", "content": assistant_text})


def get_feedback_key(message: discord.Message):
    return (message.channel.id, message.author.id)


def get_redirect_key(message: discord.Message):
    return (message.channel.id, message.author.id)


def format_user_targeted_text(message: discord.Message, content: str) -> str:
    return content


async def send_user_message(message: discord.Message, content: str, channel=None):
    safe_content = sanitize_outgoing_text(content)
    target_channel = channel or message.channel
    await send_with_typing(target_channel, format_user_targeted_text(message, safe_content))


async def send_targeted_message(message: discord.Message, channel, content: str):
    safe_content = sanitize_outgoing_text(content)
    await send_with_typing(channel, safe_content)


async def send_with_typing(channel, content: str):
    safe_content = sanitize_outgoing_text(content) if isinstance(content, str) else content
    async with channel.typing():
        await channel.send(safe_content, allowed_mentions=NO_MENTIONS)


async def send_embed_with_typing(channel, embed: discord.Embed):
    async with channel.typing():
        await channel.send(embed=embed, allowed_mentions=NO_MENTIONS)


async def send_long_interaction_reply(interaction: discord.Interaction, content: str):
    text = sanitize_outgoing_text(content or "")
    if len(text) <= 1900:
        await interaction.followup.send(text, allowed_mentions=NO_MENTIONS)
        return

    # Graceful fallback for long AI output in command responses.
    preview = text[:800] + "\n\nFull response attached as file."
    payload = io.BytesIO(text.encode("utf-8"))
    file = discord.File(payload, filename="witan-response.txt")
    await interaction.followup.send(preview, file=file, allowed_mentions=NO_MENTIONS)


async def send_intro_dm_to_user(user: discord.abc.User):
    dm = user.dm_channel or await user.create_dm()
    await send_with_typing(dm, FIRST_TIME_INTRO)
    INTRODUCED_USERS.add(user.id)


async def send_intro_dm_to_user_id(user_id: int):
    user = client.get_user(user_id)
    if not user:
        user = await client.fetch_user(user_id)
    await send_intro_dm_to_user(user)


def parse_redirect_choice(text: str):
    normalized = (text or "").strip().lower()
    mapping = {
        "1": "defender_detection",
        "one": "defender_detection",
        "2": "hardening",
        "two": "hardening",
        "3": "incident_response",
        "three": "incident_response",
    }
    return mapping.get(normalized)


def build_redirect_prompt(original_text: str, selected_track: str):
    track_instruction = {
        "defender_detection": (
            "Explain defender detection strategy: signals, logs, telemetry, detection logic, and SOC workflow. "
            "No exploit instructions."
        ),
        "hardening": (
            "Explain hardening strategy: secure configuration, prevention controls, validation checklist, and monitoring."
        ),
        "incident_response": (
            "Explain incident response strategy: triage, containment, eradication, recovery, and post-incident lessons learned."
        ),
    }[selected_track]

    return (
        "You are a cybersecurity mentor. Keep the response defensive, legal, and practical.\n\n"
        f"Topic raised by user:\n{original_text}\n\n"
        f"Requested track:\n{track_instruction}\n\n"
        "Use concise sections and actionable bullets."
    )


def apply_prompt_repetition(messages):
    enabled = PROMPT_REPETITION_ENABLED
    repeat_times = PROMPT_REPETITION_TIMES

    if not enabled:
        return messages
    if repeat_times <= 1:
        return messages
    if not messages:
        return messages

    # Copy messages so runtime prompt tuning never mutates conversation memory in-place.
    repeated = [dict(m) for m in messages]

    last_user_index = None
    for i in range(len(repeated) - 1, -1, -1):
        if repeated[i].get("role") == "user":
            last_user_index = i
            break

    if last_user_index is None:
        return repeated

    original_content = str(repeated[last_user_index].get("content", "")).strip()
    if not original_content:
        return repeated

    # Repeat only the latest user turn, matching the current "emphasis" strategy.
    repeated_content = "\n\n".join([original_content] * repeat_times)
    if len(repeated_content) > PROMPT_REPETITION_MAX_CHARS:
        return repeated

    repeated[last_user_index]["content"] = repeated_content
    return repeated


def moderate_ai_output(text: str):
    # Output sanitization/moderation disabled: return model output unchanged.
    return text, None


async def send_long_message(channel, content: str, prefix: str = ""):
    if not content:
        return

    limit = 2000
    content = sanitize_outgoing_text(content)
    if prefix:
        content = f"{prefix}{content}"
    chunks = [content[i:i + limit] for i in range(0, len(content), limit)]

    for chunk in chunks[:MAX_PUBLIC_REPLIES]:
        await send_with_typing(channel, chunk)

    if len(chunks) > MAX_PUBLIC_REPLIES:
        await send_with_typing(channel, "[response truncated]")


def get_or_create_user_lock(user_id: int):
    lock = USER_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        USER_LOCKS[user_id] = lock
    return lock


def check_rate_limit(user_id: int):
    now = time.monotonic()

    # Fast per-user cooldown to prevent burst spam.
    last_at = USER_LAST_REQUEST_AT.get(user_id, 0.0)
    if now - last_at < USER_COOLDOWN_SECONDS:
        retry_after = USER_COOLDOWN_SECONDS - (now - last_at)
        return False, f"Please wait {retry_after:.1f}s before sending the next request."

    timestamps = USER_REQUEST_TIMESTAMPS[user_id]
    # Sliding-window cap keeps sustained request volume under control.
    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        wait_seconds = RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])
        return False, f"Rate limit reached. Please try again in {max(1, int(wait_seconds))}s."

    timestamps.append(now)
    USER_LAST_REQUEST_AT[user_id] = now
    return True, None


async def regenerate_variant(original_prompt: str, prior_reply: str, mode: str):
    style = "simpler" if mode == "simpler" else "deeper and more technical"
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append(
        {
            "role": "user",
            "content": (
                "Rewrite the previous explanation for a cybersecurity learner. "
                f"Make it {style}. Keep it safe and professional.\n\n"
                f"Original user request:\n{original_prompt}\n\n"
                f"Previous explanation:\n{prior_reply}"
            ),
        }
    )
    return await query_ai(messages)


async def handle_feedback_message(message: discord.Message, text: str) -> bool:
    key = get_feedback_key(message)
    session = FEEDBACK_SESSIONS.get(key)
    if not session:
        return False

    normalized = text.strip().lower()

    if session["state"] == "awaiting_clear":
        if normalized in {"yes", "y", "clear", "got it"}:
            FEEDBACK_SESSIONS.pop(key, None)
            await send_user_message(message, "Great, glad it helped. Send your next topic whenever you're ready.")
            return True

        if normalized in {"no", "n", "not clear"}:
            session["state"] = "awaiting_mode"
            await send_user_message(message, "Got it. I can regenerate this for you. Reply `simpler` or `deeper`.")
            return True

        # Treat any other message as a new prompt and exit feedback mode quietly.
        FEEDBACK_SESSIONS.pop(key, None)
        return False

    if session["state"] == "awaiting_mode":
        if normalized not in {"simpler", "deeper"}:
            # User moved on to a new prompt; stop forcing feedback flow.
            FEEDBACK_SESSIONS.pop(key, None)
            return False

        mode = normalized
        async with AI_SEMAPHORE:
            async with message.channel.typing():
                regenerated = await regenerate_variant(session["prompt"], session["reply"], mode)

        if not regenerated:
            FEEDBACK_SESSIONS.pop(key, None)
            await send_user_message(message, "I'm having trouble reaching the AI service right now. Please try again in a moment.")
            return True

        moderated_reply, output_block_reason = moderate_ai_output(regenerated)
        if output_block_reason:
            BLOCKED_CATEGORY_COUNTS[output_block_reason] += 1

        session["reply"] = moderated_reply
        session["state"] = "awaiting_clear"

        memory_key = session["memory_key"]
        save_turn(memory_key, f"feedback:{mode}", moderated_reply)
        await send_long_message(message.channel, moderated_reply)
        await send_user_message(message, FEEDBACK_PROMPT)
        return True

    FEEDBACK_SESSIONS.pop(key, None)
    return False


async def handle_redirect_message(message: discord.Message, text: str) -> bool:
    key = get_redirect_key(message)
    session = REDIRECT_SESSIONS.get(key)
    if not session:
        return False

    # Expire stale redirect sessions.
    if time.time() - session["created_at"] > REDIRECT_SESSION_TTL_SECONDS:
        REDIRECT_SESSIONS.pop(key, None)
        return False

    choice = parse_redirect_choice(text)
    if not choice:
        # Make redirect optional: if user continues with normal text, exit redirect mode.
        if text.strip():
            REDIRECT_SESSIONS.pop(key, None)
            return False
        return False

    prompt = build_redirect_prompt(session["original_text"], choice)
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    async with AI_SEMAPHORE:
        async with message.channel.typing():
            reply = await query_ai(messages)

    if not reply:
        await send_user_message(message, "I'm having trouble reaching the AI service right now. Please try again in a moment.")
        REDIRECT_SESSIONS.pop(key, None)
        return True

    moderated_reply, output_block_reason = moderate_ai_output(reply)
    if output_block_reason:
        BLOCKED_CATEGORY_COUNTS[output_block_reason] += 1

    memory_key = get_memory_key(message)
    save_turn(memory_key, f"redirect:{choice}:{session['original_text']}", moderated_reply)
    response_channel = message.channel
    await send_long_message(response_channel, moderated_reply)

    FEEDBACK_SESSIONS[(response_channel.id, message.author.id)] = {
        "state": "awaiting_clear",
        "prompt": session["original_text"],
        "reply": moderated_reply,
        "memory_key": memory_key,
    }
    await send_targeted_message(message, response_channel, FEEDBACK_PROMPT)

    REDIRECT_SESSIONS.pop(key, None)
    return True


def _format_elapsed_readable(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        unit = "second" if seconds == 1 else "seconds"
        return f"{seconds} {unit}"
    minutes = seconds // 60
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    hours = minutes // 60
    if hours < 24:
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit}"
    days = hours // 24
    unit = "day" if days == 1 else "days"
    return f"{days} {unit}"


def _format_goal_started_at(ts: int) -> str:
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def format_goal_title_for_display(goal_text: str) -> str:
    text = str(goal_text or "")
    text = re.sub(r"<@!?(\d+)>", "", text)
    text = re.sub(r"<@&(\d+)>", "", text)
    text = re.sub(r"<#(\d+)>", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .-_")
    text = re.sub(r"^(create|make|give me|set)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(a|an)\s+", "", text, flags=re.IGNORECASE)

    lowered = text.lower()
    if "oscp" in lowered and "plan" in lowered and (
        "one week" in lowered or "1 week" in lowered or "7 day" in lowered or "seven day" in lowered
    ):
        return "One-Week OSCP Plan"
    return text[:120] if text else "Your Goal"


def build_goal_saved_message(ts: int) -> str:
    dt = datetime.fromtimestamp(int(ts))
    date_part = dt.strftime("%Y-%m-%d")
    time_part = dt.strftime("%H:%M:%S")
    return (
        "🔥 Goal saved! The journey begins now.\n"
        f"Started on: {date_part} | {time_part}\n\n"
        "Each time you finish a session OR task, Send Message:\n"
        " `Day 1 complete` \n\n"
        "I'll record your progress and keep pushing you toward mastery.\n\n"
        "Let's execute. 💪"
    )


def build_goal_saved_embed(ts: int, title: str = "Goal Saved") -> discord.Embed:
    embed = discord.Embed(
        title=title,
        description=build_goal_saved_message(ts),
        color=discord.Color.blurple(),
    )
    return embed



async def run_user_reminder_scan():
    now_ts = int(time.time())
    due = [r for r in USER_REMINDERS if not r.get("sent") and int(r.get("remind_at", 0)) <= now_ts]
    if not due:
        return

    for reminder in due:
        user_id = int(reminder["user_id"])
        user = client.get_user(user_id)
        if not user:
            try:
                user = await client.fetch_user(user_id)
            except Exception:
                continue

        try:
            dm = user.dm_channel or await user.create_dm()
            embed = discord.Embed(
                title="This is Your Reminder",
                description=str(reminder.get("message", "")),
                color=discord.Color.yellow(),
            )
            await dm.send(embed=embed, allowed_mentions=NO_MENTIONS)
            reminder["sent"] = True
        except Exception:
            logging.exception("Failed to send user reminder DM")


def _looks_like_completed_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Do not treat negated phrases as completion.
    if _looks_like_not_completed_reply(t):
        return False
    return bool(re.search(r"\b(completed|complete|done)\b", t))


def _looks_like_not_completed_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"\bnot\s+complete(?:d)?\b",
        r"\bdid\s+not\s+complete\b",
        r"\bdidn['’]t\s+complete\b",
        r"\bincomplete\b",
        r"\bnot\s+yet\b",
        r"\bpending\b",
        r"^\s*no\b",
    ]
    return any(re.search(p, t) for p in patterns)


def parse_day_completion_number(text: str):
    t = (text or "").strip().lower()
    if not t:
        return None
    m = re.search(r"\bday\s*([1-9]\d{0,2})\s*(?:is\s*)?(?:complete|completed|done)\b", t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


async def run_goal_reminder_scan():
    now_ts = int(time.time())
    due_items = await get_due_goal_reminders(now_ts)
    if not due_items:
        return

    for item in due_items:
        reminder_key = (item["goal_id"], item["day_number"])
        if reminder_key in SENT_DAY_REMINDERS:
            continue

        user = client.get_user(item["user_id"])
        if not user:
            try:
                user = await client.fetch_user(item["user_id"])
            except Exception:
                continue

        try:
            dm = user.dm_channel or await user.create_dm()
            embed = discord.Embed(
                title="Yo quick reminder",
                description=f"Day {item['day_number']} is almost complete.",
                color=discord.Color.blurple(),
            )
            embed.add_field(
                name="Check-in",
                value=f"Have you completed day {item['day_number']} task?",
                inline=False,
            )
            embed.set_footer(text="Reply with completed or not completed.")
            await dm.send(embed=embed, allowed_mentions=NO_MENTIONS)
            SENT_DAY_REMINDERS.add(reminder_key)
            PENDING_DAY_CHECKS[item["user_id"]] = {
                "goal_id": item["goal_id"],
                "day_number": item["day_number"],
                "created_at": now_ts,
            }
        except Exception:
            logging.exception("Failed to send goal reminder DM")


async def goal_reminder_worker():
    while True:
        try:
            await run_goal_reminder_scan()
        except Exception:
            logging.exception("Goal reminder worker error")
        await asyncio.sleep(max(60, GOAL_REMINDER_INTERVAL_SECONDS))


async def user_reminder_worker():
    while True:
        try:
            await run_user_reminder_scan()
        except Exception:
            logging.exception("User reminder worker error")
        await asyncio.sleep(max(3, USER_REMINDER_SCAN_INTERVAL_SECONDS))


async def run_news_digest_scan():
    now_local = datetime.now()
    due = await get_due_news_subscriptions(now_local)
    if not due:
        return

    embed_by_topic = {}
    for sub in due:
        user_id = int(sub["user_id"])
        topic_query = (sub.get("topic_query") or "").strip() or NEWS_DEFAULT_TOPIC
        user = client.get_user(user_id)
        if not user:
            try:
                user = await client.fetch_user(user_id)
            except Exception:
                continue

        try:
            embed = embed_by_topic.get(topic_query)
            if embed is None:
                embed = await build_daily_news_embed(topic_query)
                embed_by_topic[topic_query] = embed
            if embed is None:
                logging.warning("News digest: no matching NewsAPI items for topic=%s", topic_query)
                dm = user.dm_channel or await user.create_dm()
                no_match_embed = discord.Embed(
                    title="🔐 Daily Cybersecurity News",
                    description=(
                        f"No matching news found today for topic: `{topic_query}`.\n"
                        "Try a broader topic with `/onnews`, for example: `cybersecurity`, `malware`, `CVE`."
                    ),
                    color=discord.Color.yellow(),
                )
                await dm.send(embed=no_match_embed, allowed_mentions=NO_MENTIONS)
                await mark_news_subscription_sent(user_id, now_local.strftime("%Y-%m-%d"))
                continue
            dm = user.dm_channel or await user.create_dm()
            await dm.send(embed=embed, allowed_mentions=NO_MENTIONS)
            await mark_news_subscription_sent(user_id, now_local.strftime("%Y-%m-%d"))
        except Exception:
            logging.exception("Failed to send daily news digest DM to user %s", user_id)


async def news_digest_worker():
    while True:
        try:
            await run_news_digest_scan()
        except Exception:
            logging.exception("News digest worker error")
        await asyncio.sleep(max(15, NEWS_DIGEST_SCAN_INTERVAL_SECONDS))


async def handle_goal_reminder_reply(message: discord.Message, text: str) -> bool:
    user_id = message.author.id
    pending_check = PENDING_DAY_CHECKS.get(user_id)
    if not pending_check:
        # Also support direct progress updates without reminder prompt, e.g. "day1complete".
        day_number = parse_day_completion_number(text)
        if day_number is None:
            return False

        result = await record_goal_day_completion(user_id, day_number)
        status = result.get("status")
        if status == "no_goal":
            embed = discord.Embed(
                title="Progress Update",
                description="No active goal found. Set a goal first, then track progress with `day 1 complete`.",
                color=discord.Color.orange(),
            )
            await send_embed_with_typing(message.channel, embed)
            return True
        if status == "already_done":
            embed = discord.Embed(
                title="Progress Update",
                description=f"Day {day_number} is already marked as complete.",
                color=discord.Color.blurple(),
            )
            await send_embed_with_typing(message.channel, embed)
            return True

        done_title = "Day Recorded ✅"
        if result.get("completed_goal"):
            done_title = "Goal Completed ✅"
        embed = discord.Embed(
            title=done_title,
            description=f"I marked `Day {day_number}` as complete.",
            color=discord.Color.green(),
        )
        embed.add_field(
            name="Progress",
            value=f"{result.get('completed_count', 0)}/{result.get('duration_days', 0)} days",
            inline=True,
        )
        await send_embed_with_typing(message.channel, embed)
        return True

    if int(time.time()) - int(pending_check.get("created_at", 0)) > GOAL_REMINDER_REPLY_TTL_SECONDS:
        PENDING_DAY_CHECKS.pop(user_id, None)
        return False

    if _looks_like_not_completed_reply(text):
        PENDING_DAY_CHECKS.pop(user_id, None)
        embed = discord.Embed(
            title="Check-in Logged",
            description="No worries. Keep going and check in after your next session.",
            color=discord.Color.blurple(),
        )
        await send_embed_with_typing(message.channel, embed)
        return True

    if _looks_like_completed_reply(text):
        result = await record_goal_day_completion(user_id, int(pending_check["day_number"]))
        PENDING_DAY_CHECKS.pop(user_id, None)
        status = result.get("status")
        if status == "already_done":
            embed = discord.Embed(
                title="Progress Update",
                description=f"Day {pending_check['day_number']} is already marked as complete.",
                color=discord.Color.blurple(),
            )
            await send_embed_with_typing(message.channel, embed)
            return True

        done_title = "Day Recorded ✅"
        if result.get("completed_goal"):
            done_title = "Goal Completed ✅"
        embed = discord.Embed(
            title=done_title,
            description=f"I marked day {pending_check['day_number']} as complete.",
            color=discord.Color.green(),
        )
        embed.add_field(
            name="Progress",
            value=f"{result.get('completed_count', 0)}/{result.get('duration_days', 0)} days",
            inline=True,
        )
        await send_embed_with_typing(message.channel, embed)
        return True

    return False


def _set_goal_confirmation_payload_for_channels(message: discord.Message, response_channel, payload: dict):
    PENDING_GOAL_CONFIRMATIONS[(response_channel.id, message.author.id)] = payload
    if response_channel.id != message.channel.id:
        PENDING_GOAL_CONFIRMATIONS[(message.channel.id, message.author.id)] = payload


async def offer_goal_save_confirmation(
    message: discord.Message,
    response_channel,
    goal_title: str,
    plan_text: str = None,
    duration_days: int = 7,
    require_confirmation_if_no_active: bool = True,
):
    active_goal = await get_active_goal_snapshot_for_user(message.author.id)
    payload = {
        "created_at": time.time(),
        "owner_user_id": message.author.id,
        "goal_title": (goal_title or "")[:300],
        "plan_text": (plan_text or "")[:4000] if plan_text else None,
        "duration_days": max(1, min(365, int(duration_days))),
    }

    if active_goal:
        payload["action"] = "replace_active_goal"
        payload["existing_goal_id"] = int(active_goal["goal_id"])
        _set_goal_confirmation_payload_for_channels(message, response_channel, payload)

        started_at_text = _format_goal_started_at(active_goal["created_at"])
        progress_text = f"day {active_goal['progressed_days']}/{active_goal['duration_days']}"
        embed = discord.Embed(
            title="Yo quick reminder",
            description="You already have an active goal.",
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Current goal", value=str(active_goal["goal_text"])[:1024], inline=False)
        embed.add_field(name="Set on", value=started_at_text, inline=True)
        embed.add_field(name="Progress", value=progress_text, inline=True)
        embed.set_footer(text="Would you like me to discard this goal and set your new one? Reply `yes` or `no`.")
        await send_embed_with_typing(response_channel, embed)
        return

    if not require_confirmation_if_no_active:
        await save_goal_to_db(
            message,
            goal_text=payload["goal_title"],
            plan_text=payload["plan_text"],
            duration_days=payload["duration_days"],
        )
        ts = int(time.time())
        await send_targeted_message(
            message,
            response_channel,
            build_goal_saved_message(ts),
        )
        return

    payload["action"] = "save_new_goal"
    _set_goal_confirmation_payload_for_channels(message, response_channel, payload)
    embed = discord.Embed(
        description="Would you like me to add this as your goal? Reply `yes` or `no`.",
        color=discord.Color.yellow(),
    )
    await send_embed_with_typing(response_channel, embed)


async def handle_goal_confirmation_message(message: discord.Message, text: str) -> bool:
    def _clear_matching_pending(target):
        to_delete = [k for k, v in PENDING_GOAL_CONFIRMATIONS.items() if v is target]
        for k in to_delete:
            PENDING_GOAL_CONFIRMATIONS.pop(k, None)

    key = get_goal_confirmation_key(message)
    pending = PENDING_GOAL_CONFIRMATIONS.get(key)
    if not pending:
        return False

    if time.time() - pending["created_at"] > GOAL_CONFIRMATION_TTL_SECONDS:
        _clear_matching_pending(pending)
        return False

    normalized = (text or "").strip().lower()
    action = pending.get("action", "save_new_goal")
    if int(pending.get("owner_user_id", message.author.id)) != message.author.id:
        return False

    if normalized in {"yes", "y"}:
        try:
            if action == "replace_active_goal":
                await reset_active_goal_for_user(message.author.id)

            await save_goal_to_db(
                message,
                goal_text=pending["goal_title"],
                plan_text=pending["plan_text"],
                duration_days=pending["duration_days"],
            )
            ts = int(time.time())
            if action == "replace_active_goal":
                embed = build_goal_saved_embed(ts, title="Previous goal discarded. Replaced with your new goal.")
                await send_embed_with_typing(message.channel, embed)
            else:
                embed = build_goal_saved_embed(ts)
                await send_embed_with_typing(message.channel, embed)
        except Exception:
            logging.exception("Goal confirmation save failed")
            await send_user_message(message, "I couldn't save that goal right now. Please try again.")
        finally:
            _clear_matching_pending(pending)
        return True

    if normalized in {"no", "n"}:
        _clear_matching_pending(pending)
        if action == "replace_active_goal":
            await send_user_message(message, "Understood. I kept your current goal unchanged.")
        else:
            await send_user_message(message, "No problem. I won't save this as a goal right now.")
        return True

    # Any other text means user moved on.
    _clear_matching_pending(pending)
    return False


async def process_ai_request(message: discord.Message, text: str, response_channel=None):
    user_id = message.author.id
    response_channel = response_channel or message.channel

    # Backpressure: don't allow unbounded queue growth per user.
    if USER_PENDING_COUNTS[user_id] >= MAX_USER_QUEUE:
        await send_user_message(
            message,
            "You already have a few requests queued. Please wait for earlier replies to finish.",
            channel=response_channel,
        )
        return

    USER_PENDING_COUNTS[user_id] += 1
    queued_before = USER_PENDING_COUNTS[user_id] - 1

    if queued_before > 0:
        await send_user_message(
            message,
            f"Your request is in queue. Current position: {queued_before + 1}.",
            channel=response_channel,
        )

    # Serialize each user's requests so replies stay ordered and memory remains coherent.
    lock = get_or_create_user_lock(user_id)

    try:
        async with lock:
            allowed, reason = check_rate_limit(user_id)
            if not allowed:
                await send_user_message(message, reason, channel=response_channel)
                return

            memory_key = ("dm", response_channel.id, message.author.id)
            update_learner_profile(user_id, text)
            messages = build_messages_with_memory(memory_key, text)

            async with AI_SEMAPHORE:
                async with response_channel.typing():
                    reply = await query_ai(messages)

            if not reply:
                await send_user_message(
                    message,
                    "I'm having trouble reaching the AI service right now. Please try again in a moment.",
                    channel=response_channel,
                )
                return

            moderated_reply, output_block_reason = moderate_ai_output(reply)
            if output_block_reason:
                BLOCKED_CATEGORY_COUNTS[output_block_reason] += 1

            save_turn(memory_key, text, moderated_reply)
            await send_long_message(response_channel, moderated_reply)

            if should_offer_goal_confirmation(text):
                await offer_goal_save_confirmation(
                    message,
                    response_channel,
                    goal_title=text[:300],
                    plan_text=moderated_reply[:4000],
                    duration_days=7,
                )

            if should_offer_feedback(text):
                feedback_key = (response_channel.id, message.author.id)
                FEEDBACK_SESSIONS[feedback_key] = {
                    "state": "awaiting_clear",
                    "prompt": text,
                    "reply": moderated_reply,
                    "memory_key": memory_key,
                }
                await send_targeted_message(message, response_channel, FEEDBACK_PROMPT)
    finally:
        USER_PENDING_COUNTS[user_id] = max(0, USER_PENDING_COUNTS[user_id] - 1)


# ---------------- AI QUERY (STREAMING MATCH) ----------------
async def query_ai(messages):
    """
    Streaming-enabled call to match playground behavior.
    Returns full combined text exactly as model generated.
    """
    try:
        loop = asyncio.get_running_loop()
        model_messages = apply_prompt_repetition(messages)

        def stream_call():
            stream = client_ai.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=model_messages,
                temperature=NVIDIA_TEMPERATURE,
                top_p=NVIDIA_TOP_P,
                max_tokens=NVIDIA_MAX_TOKENS,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True,
                timeout=NVIDIA_REQUEST_TIMEOUT,
            )

            parts = []
            for event in stream:
                try:
                    choice = event.choices[0]
                    delta = getattr(choice, "delta", None)

                    chunk = None
                    if delta and isinstance(delta, dict):
                        chunk = delta.get("content")
                    elif delta:
                        chunk = getattr(delta, "content", None)

                    if chunk:
                        parts.append(chunk)
                except Exception:
                    continue

            return "".join(parts).strip()

        return await asyncio.wait_for(
            loop.run_in_executor(None, stream_call),
            timeout=NVIDIA_REQUEST_TIMEOUT + 5,
        )

    except asyncio.TimeoutError:
        logging.warning("AI timeout")
        return None
    except Exception:
        logging.exception("AI error")
        return None



def _noop_decorator(*args, **kwargs):
    def _inner(func):
        return func
    return _inner


_allowed_installs_decorator = getattr(app_commands, "allowed_installs", _noop_decorator)
_allowed_contexts_decorator = getattr(app_commands, "allowed_contexts", _noop_decorator)


@COMMAND_TREE.command(name="scanurl", description="Scan a URL with multiple threat-intel providers")
@app_commands.describe(url="URL to scan (must start with http:// or https://)")
@_allowed_installs_decorator(guilds=True, users=True)
@_allowed_contexts_decorator(guilds=True, dms=True, private_channels=True)
async def slash_scanurl(interaction: discord.Interaction, url: str):
    # Manual URL scanner for debugging and moderation checks.
    candidate = (url or "").strip()
    if not re.match(r"^https?://", candidate, re.IGNORECASE):
        await interaction.response.send_message("Please provide a valid URL starting with http:// or https://", ephemeral=True)
        return

    if interaction.guild is not None:
        perms = getattr(interaction.user, "guild_permissions", None)
        if not perms or not getattr(perms, "manage_guild", False):
            await interaction.response.send_message("Only server admins/moderators can use /scanurl in servers.", ephemeral=True)
            return

    await interaction.response.defer(thinking=True, ephemeral=True)

    if is_url_allowlisted(candidate):
        await interaction.followup.send(
            f"This URL is allowlisted and skipped: {candidate}",
            ephemeral=True,
            allowed_mentions=NO_MENTIONS,
        )
        return

    try:
        result = await scan_single_url(candidate)
    except Exception:
        logging.exception("slash_scanurl failed for url=%s", candidate)
        await interaction.followup.send("URL scan failed. Please try again.", ephemeral=True, allowed_mentions=NO_MENTIONS)
        return

    requested_by = str(interaction.user)
    embed = build_url_scan_result_embed(result, requested_by=requested_by)
    await interaction.followup.send(embed=embed, ephemeral=True, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="goal", description="Show your active goal")
@_allowed_installs_decorator(guilds=True, users=True)
@_allowed_contexts_decorator(guilds=True, dms=True, private_channels=True)
async def slash_goal(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    active_goal = await get_active_goal_for_user(interaction.user.id)
    if not active_goal:
        embed = discord.Embed(
            title="Your Goal",
            description="You don't have an active goal yet.",
            color=discord.Color.blurple(),
        )
        await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)
        return

    created_at = int(active_goal["created_at"])
    since = _format_elapsed_readable(max(0, int(time.time()) - created_at))
    goal_title = format_goal_title_for_display(active_goal["goal_text"])
    embed = discord.Embed(title="Your Goal", color=discord.Color.blurple())
    embed.add_field(name="Goal", value=goal_title, inline=False)
    embed.add_field(name="Duration", value=f"{active_goal['duration_days']} days", inline=True)
    embed.add_field(name="Started", value=f"{since} ago", inline=True)
    await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="resetgoal", description="Reset your active goal")
@_allowed_installs_decorator(guilds=True, users=True)
@_allowed_contexts_decorator(guilds=True, dms=True, private_channels=True)
async def slash_resetgoal(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    reset_result = await reset_active_goal_for_user(interaction.user.id)
    if reset_result.get("status") != "ok":
        embed = discord.Embed(
            title="Goal Reset",
            description="No active goal found.",
            color=discord.Color.orange(),
        )
        await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)
        return

    goal_id = int(reset_result.get("goal_id", 0) or 0)
    if goal_id:
        to_remove = {k for k in SENT_DAY_REMINDERS if int(k[0]) == goal_id}
        SENT_DAY_REMINDERS.difference_update(to_remove)
    PENDING_DAY_CHECKS.pop(interaction.user.id, None)

    embed = discord.Embed(
        title="Goal Reset ✅",
        description="Your active goal has been reset.",
        color=discord.Color.green(),
    )
    await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="reminders", description="View your active reminders")
@_allowed_installs_decorator(guilds=False, users=True)
@_allowed_contexts_decorator(guilds=False, dms=True, private_channels=True)
async def slash_reminders(interaction: discord.Interaction):
    if interaction.guild is not None:
        await interaction.response.send_message("Please use this command in DM with the bot.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    user_items = [r for r in USER_REMINDERS if int(r.get("user_id", 0)) == user_id and not r.get("sent")]
    user_items.sort(key=lambda r: int(r.get("remind_at", 0)))

    if not user_items:
        embed = discord.Embed(
            title="Your Reminders",
            description="You don't have any active reminders.",
            color=discord.Color.yellow(),
        )
        await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)
        return

    lines = []
    for idx, item in enumerate(user_items[:10], start=1):
        remind_at = datetime.fromtimestamp(int(item["remind_at"])).strftime("%Y-%m-%d %H:%M:%S")
        msg = str(item.get("message", "")).replace("\n", " ").strip()
        if len(msg) > 80:
            msg = msg[:77] + "..."
        lines.append(f"{idx}. {remind_at} - {msg}")

    extra = len(user_items) - 10
    if extra > 0:
        lines.append(f"...and {extra} more.")

    embed = discord.Embed(
        title="Your Reminders",
        description="\n".join(lines),
        color=discord.Color.yellow(),
    )
    await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="setreminder", description="Create a reminder")
@app_commands.rename(reminder_time="time")
@app_commands.describe(
    date="Date (e.g., 24/02/2026)",
    reminder_time="Time (e.g., 4:55 PM)",
    message="Reminder message",
)
@_allowed_installs_decorator(guilds=False, users=True)
@_allowed_contexts_decorator(guilds=False, dms=True, private_channels=True)
async def slash_setreminder(interaction: discord.Interaction, date: str, reminder_time: str, message: str):
    if interaction.guild is not None:
        await interaction.response.send_message("Please use this command in DM with the bot.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)
    date_time = f"{(date or '').strip()} {(reminder_time or '').strip()}".strip()
    dt = parse_reminder_datetime_input(date_time)
    if not dt:
        embed = discord.Embed(
            title="Invalid Date/Time",
            description="Try: 24/02/2026 4:55 PM",
            color=discord.Color.yellow(),
        )
        await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)
        return

    ts = int(dt.timestamp())
    if ts <= int(time.time()):
        embed = discord.Embed(
            title="Invalid Date/Time",
            description="That time is in the past. Please send a future date/time.",
            color=discord.Color.yellow(),
        )
        await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)
        return

    global USER_REMINDER_NEXT_ID
    reminder_id = USER_REMINDER_NEXT_ID
    USER_REMINDER_NEXT_ID += 1
    USER_REMINDERS.append(
        {
            "id": reminder_id,
            "user_id": interaction.user.id,
            "remind_at": ts,
            "message": (message or "").strip()[:1500],
            "created_at": int(time.time()),
            "sent": False,
        }
    )

    when_text = format_reminder_when_text(ts)
    embed = discord.Embed(
        title="🎉 All set!",
        description=f"I'll remind you {when_text} to:\n\"{(message or '').strip()[:1024]}\"",
        color=discord.Color.yellow(),
    )
    await interaction.followup.send(embed=embed, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="onnews", description="Enable daily cybersecurity news digest in DM")
@app_commands.rename(time_input="time")
@app_commands.describe(
    time_input="Daily time (e.g., 20:30 or 8:30 PM)",
    topic="Interested topic (e.g., ransomware, CVE, phishing)",
)
@_allowed_installs_decorator(guilds=True, users=True)
@_allowed_contexts_decorator(guilds=True, dms=True, private_channels=True)
async def slash_onnews(interaction: discord.Interaction, time_input: str, topic: str):
    is_dm = interaction.guild is None
    if not NEWS_API_KEY:
        await interaction.response.send_message(
            "News digest is unavailable right now because `News_API` is not configured.",
            ephemeral=not is_dm,
            allowed_mentions=NO_MENTIONS,
        )
        return
    parsed = parse_daily_time_input(time_input)
    if not parsed:
        await interaction.response.send_message(
            "Invalid time format. Try `08:30` or `8:30 PM`.",
            ephemeral=not is_dm,
            allowed_mentions=NO_MENTIONS,
        )
        return
    topic_value = (topic or "").strip()[:200]
    if not topic_value:
        await interaction.response.send_message(
            "Please provide a topic, for example: `CVE`, `malware`, `phishing`, `cloud security`.",
            ephemeral=not is_dm,
            allowed_mentions=NO_MENTIONS,
        )
        return

    hour, minute = parsed
    await interaction.response.defer(thinking=True, ephemeral=not is_dm)
    await upsert_news_subscription(interaction.user.id, hour, minute, topic_value)
    display = datetime(2000, 1, 1, hour, minute).strftime("%I:%M %p").lstrip("0")
    embed = discord.Embed(
        title="News Digest Enabled",
        description=(
            f"You will receive 4 summarized cybersecurity news items daily at `{display}` "
            "(server local time).\n"
            f"Topic: `{topic_value}`"
        ),
        color=discord.Color.yellow(),
    )
    embed.set_footer(text="Source: NewsAPI")
    await interaction.followup.send(embed=embed, ephemeral=not is_dm, allowed_mentions=NO_MENTIONS)


@COMMAND_TREE.command(name="offnews", description="Disable daily cybersecurity news digest")
@_allowed_installs_decorator(guilds=True, users=True)
@_allowed_contexts_decorator(guilds=True, dms=True, private_channels=True)
async def slash_offnews(interaction: discord.Interaction):
    is_dm = interaction.guild is None
    await interaction.response.defer(thinking=True, ephemeral=not is_dm)
    await disable_news_subscription(interaction.user.id)
    embed = discord.Embed(
        title="News Digest Disabled",
        description="Daily cybersecurity news DM is turned off.",
        color=discord.Color.yellow(),
    )
    await interaction.followup.send(embed=embed, ephemeral=not is_dm, allowed_mentions=NO_MENTIONS)


# ---------------- EVENTS ----------------
@client.event
async def on_ready():
    # Startup tasks: set loop, init DB, start workers, and sync slash commands once.
    global USER_REMINDER_TASK, GOAL_REMINDER_TASK, NEWS_DIGEST_TASK, BOT_LOOP, APP_COMMANDS_SYNCED
    print(f"Connected as {client.user} (ID: {client.user.id})")

    activity = None
    await client.change_presence(status=discord.Status.online, activity=activity)
    BOT_LOOP = asyncio.get_running_loop()
    log_url_scan_provider_config()
    try:
        await run_provider_self_test()
    except Exception:
        logging.exception("URL scan provider self-test failed")
    try:
        await asyncio.to_thread(init_goals_db)
    except Exception:
        logging.exception("Goals DB init failed")
    if GOAL_REMINDER_TASK is None or GOAL_REMINDER_TASK.done():
        GOAL_REMINDER_TASK = asyncio.create_task(goal_reminder_worker())
    if USER_REMINDER_TASK is None or USER_REMINDER_TASK.done():
        USER_REMINDER_TASK = asyncio.create_task(user_reminder_worker())
    if NEWS_DIGEST_TASK is None or NEWS_DIGEST_TASK.done():
        NEWS_DIGEST_TASK = asyncio.create_task(news_digest_worker())
    if not APP_COMMANDS_SYNCED:
        try:
            synced = await COMMAND_TREE.sync()
            APP_COMMANDS_SYNCED = True
            logging.info("Synced %s global app commands", len(synced))
            if synced:
                logging.info("Global commands: %s", ", ".join(sorted(cmd.name for cmd in synced)))
            # Also sync per-guild for faster command visibility during rollout.
            for guild in client.guilds:
                try:
                    COMMAND_TREE.clear_commands(guild=guild)
                    COMMAND_TREE.copy_global_to(guild=guild)
                    guild_synced = await COMMAND_TREE.sync(guild=guild)
                    logging.info("Synced %s app commands for guild %s (%s)", len(guild_synced), guild.name, guild.id)
                    if guild_synced:
                        logging.info(
                            "Guild %s commands: %s",
                            guild.id,
                            ", ".join(sorted(cmd.name for cmd in guild_synced)),
                        )
                except Exception:
                    logging.exception("Failed guild app-command sync for guild id=%s", guild.id)
        except Exception:
            logging.exception("Failed to sync app commands")
    logging.info(f"Logged in as {client.user}")


@client.event
async def on_member_join(member: discord.Member):
    # Optional onboarding: DM intro to each new human member.
    if not WELCOME_DM_ON_MEMBER_JOIN:
        return
    if member.bot:
        return

    try:
        await send_intro_dm_to_user(member)
    except discord.Forbidden:
        logging.info("Could not send join DM to user %s (DMs closed).", member.id)
    except Exception:
        logging.exception("Failed to send join DM to user %s", member.id)


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    text = (message.content or "").strip()
    if not text:
        return

    # URL safety scan runs in both DM and server contexts.
    await scan_message_links_and_warn(message, text)

    # Goal progress updates/check-ins should work in both DM and server.
    if await handle_goal_reminder_reply(message, text):
        return

    # Server behavior:
    # - Slash commands handle goal/reminder actions.
    # - Mention works for normal Q&A.
    if message.guild:
        # Allow confirmation follow-ups (yes/no) in server without requiring a mention.
        if await handle_goal_confirmation_message(message, text):
            return

        mentioned_bot = client.user is not None and client.user in message.mentions
        if mentioned_bot:
            mention_stripped = re.sub(rf"<@!?{client.user.id}>", "", text).strip() if client.user else text.strip()

            # Send intro only once per user when they first greet/ping in server.
            if message.author.id not in INTRODUCED_USERS and (not mention_stripped or is_greeting(mention_stripped)):
                INTRODUCED_USERS.add(message.author.id)
                try:
                    await send_with_typing(message.channel, GREETING_REPLY)
                except Exception:
                    logging.exception("Failed to send server mention intro for user %s", message.author.id)
                return

            if not mention_stripped:
                await send_with_typing(message.channel, "I'm here. Mention me with your question.")
                return

            await process_ai_request(message, mention_stripped, response_channel=message.channel)
        return

    # Everything below is DM-only logic.
    # Order matters: reminder handlers first, then moderation/fast-paths, then AI.

    # Handle confirmation for auto-detected plan goals.
    if await handle_goal_confirmation_message(message, text):
        return

    # Handle post-response feedback before mention checks.
    if await handle_feedback_message(message, text):
        return

    # Handle 1/2/3 defensive learning track after blocked unsafe prompts.
    if await handle_redirect_message(message, text):
        return

    # Safety intent + keyword filter
    blocked_category = detect_safety_intent(text)
    blocked_keyword = contains_safety_keyword(text)

    if blocked_category or blocked_keyword:
        try:
            await message.delete()
        except Exception:
            pass

        reason = blocked_category or "unsafe_content"
        if blocked_keyword:
            reason = f"{reason}:{blocked_keyword}"

        BLOCKED_CATEGORY_COUNTS[reason] += 1

        safe_alt = safe_alternative_for(blocked_category or "")
        await send_user_message(message, f"[blocked] {safe_alt}")

        if blocked_category in {"malware", "intrusion"} or blocked_keyword:
            REDIRECT_SESSIONS[get_redirect_key(message)] = {
                "created_at": time.time(),
                "original_text": text,
            }

        return

    # Profanity
    if is_profane(text):
        try:
            await message.delete()
        except Exception:
            pass

        await send_user_message(message, "[warning] Please keep it respectful and professional.")
        return

    # Fast-path date/time answers from local system clock (no AI request).
    datetime_reply = get_datetime_reply(text)
    if datetime_reply:
        await send_user_message(message, datetime_reply)
        return

    # Fast-path greeting response (no AI call).
    if is_greeting(text):
        INTRODUCED_USERS.add(message.author.id)
        await send_user_message(message, GREETING_REPLY)
        return

    # One-time intro for first non-greeting message from each user.
    if message.author.id not in INTRODUCED_USERS:
        try:
            await send_intro_dm_to_user(message.author)
        except Exception:
            await send_user_message(message, FIRST_TIME_INTRO)

    await process_ai_request(message, text)


# ---------------- START ----------------
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)





