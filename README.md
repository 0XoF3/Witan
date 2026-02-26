
# 🛡️ Witan — Your Cybersecurity Mentor Bot

Witan is an AI-powered Discord bot designed to mentor, guide, and protect cybersecurity learners and communities.

It combines **structured learning guidance**, **automated threat intelligence**, and **real-time URL safety scanning** into a single assistant.

It showes offline but it is online 🟢 <img width="228" height="59" alt="image" src="https://github.com/user-attachments/assets/34af890f-4d64-41a9-ab9d-1a24b3f095e2" />

👉 **Install Witan:**
[https://witan.webapp-1c9.workers.dev/install](https://witan.webapp-1c9.workers.dev/install)

---

## 🔎 What Witan Does

Witan operates in three core domains:

### 1️⃣ Cybersecurity Mentorship (AI-Powered)

* Personalized learning roadmaps (e.g., OSCP, Security+, Bug Bounty, SOC)
* Structured one-week study plans
* Goal tracking with progress logging
* Skill-level detection (Beginner / Intermediate / Advanced)
* Defensive explanations of attack techniques
* Regenerated explanations (simpler / deeper versions)
* Certification preparation guidance
* Real-world attack & defense breakdowns (safe, legal framing only)

Witan behaves like a structured cybersecurity coach — not a random Q&A bot.

---

### 2️⃣ Real-Time URL Threat Intelligence Scanning

Whenever a link is posted:

* Automatically scans URLs using:

  * Google Safe Browsing
  * VirusTotal
  * AlienVault OTX
* Aggregates provider results into a risk score (0–100%)
* Classifies risk level:

  * SAFE
  * LIKELY SAFE
  * SUSPICIOUS
  * UNSAFE
* Shows provider coverage and detection summaries
* Caches results to reduce redundant API calls
* Supports manual `/scanurl` command for moderators

This reduces phishing exposure and malicious link risks inside Discord communities.

---

### 3️⃣ Daily Cybersecurity News Digest

Users can subscribe to:

* 4 curated cybersecurity news summaries per day
* Topic-based filtering (e.g., CVE, ransomware, phishing)
* English-only filtering
* Cyber relevance validation
* AI-generated concise summaries
* DM delivery at a chosen time

Powered by NewsAPI + AI summarization.

Commands:

* `/onnews`
* `/offnews`

---

### 4️⃣ Goal Tracking & Accountability System

Users can:

* Set learning goals
* Generate 7-day plans
* Log progress with:
  `Day 1 complete`
* Receive automated reminders before daily rollover
* Reset goals with `/resetgoal`
* View active goal with `/goal`

Witan tracks progress in SQLite and manages structured accountability.

---

### 5️⃣ Reminder System (DM Only)

Users can:

* Set custom reminders with `/setreminder`
* View active reminders with `/reminders`
* Receive scheduled DM notifications

---

## 👥 Who Witan Helps

Witan is designed for:

### 🔰 Beginners

* Structured starting roadmap
* Simplified explanations
* Security mindset development

### ⚔️ Intermediate Learners

* OSCP / CEH / Security+ prep
* Lab guidance
* Exploit theory (defensive focus)

### 🎯 Bug Bounty Hunters

* Vulnerability concept breakdown
* Defensive detection perspective
* Structured skill-building

### 🛡️ SOC Analysts / Blue Teamers

* Detection logic explanation
* Incident response workflow breakdown
* Hardening guidance

### 🧠 Cybersecurity Communities

* Automatic link scanning
* Reduced phishing exposure
* Moderator visibility into threats
* Curated security news

---

## 🤖 How AI Is Used

Witan uses AI in multiple layers:

### 1. Instruction-Tuned LLM (NVIDIA NIM)

Model:

```
mistralai/mistral-7b-instruct-v0.3
```

Used for:

* Generating learning plans
* Explaining security concepts
* Summarizing news articles
* Rewriting responses (simpler / deeper)
* Defensive-only redirect explanations
* Context-aware conversation memory

Streaming responses ensure smooth delivery.

---

### 2. Contextual Memory System

* Stores recent conversation turns
* Maintains token budget
* Injects learner profile context (level, goal, timeline)
* Preserves structured continuity

---

### 3. Safety & Intent Detection Layer

Before AI is called, Witan:

* Detects exploit-generation attempts
* Blocks malicious payload requests
* Redirects unsafe intent toward:

  * Detection strategy
  * Hardening strategy
  * Incident response workflow
* Filters profanity
* Applies keyword pattern detection

This ensures legal, defensive-only guidance.

---

### 4. AI News Summarization

For each cybersecurity article:

* Validates domain & cyber relevance
* Filters non-security content
* Summarizes to 1–2 concise defensive sentences
* Limits character length
* Delivers structured embed output

---

### 5. Threat Intelligence Aggregation Engine

Not AI-based — but algorithmic scoring:

* Weighted risk scoring per provider
* Threshold-based classification
* Provider coverage transparency
* Cache-based optimization

---

## 🛠️ Slash Commands

| Command        | Description                     |
| -------------- | ------------------------------- |
| `/scanurl`     | Scan a URL manually             |
| `/goal`        | View active goal                |
| `/resetgoal`   | Reset active goal               |
| `/setreminder` | Create reminder (DM only)       |
| `/reminders`   | View reminders (DM only)        |
| `/onnews`      | Enable daily cybersecurity news |
| `/offnews`     | Disable news digest             |

---

## 🔐 Safety Architecture

Witan includes:

* Regex-based exploit detection
* Intent classification rules
* AI redirection workflow
* Rate limiting per user
* Concurrency semaphore control
* Mention sanitization
* URL allowlisting
* Output moderation hooks
* Cooldown + sliding window throttling

---

## 🗄️ Storage

SQLite database stores:

* Goals
* Goal progress
* News subscriptions
* Reminder metadata

WAL mode enabled for concurrency safety.

---

## 🚀 Installation

Add Witan to your Discord server:

👉 [https://witan.webapp-1c9.workers.dev/install](https://witan.webapp-1c9.workers.dev/install)

Then:

* Mention the bot in a server to start Q&A
* Or DM it directly for mentorship mode
* Use `/` commands for structured features

---

## 🧠 Design Philosophy

Witan is not a generic chatbot.

It is designed to:

* Teach systematically
* Enforce discipline
* Encourage defensive security thinking
* Reduce unsafe behavior
* Protect communities from malicious links
* Provide structured growth paths

It combines mentorship, threat intelligence, and automation into a single assistant.

