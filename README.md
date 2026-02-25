# Witan Bot (GitHub Ready)

## Included
- `Witan.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `render.yaml`
- `discloud.config` (optional, only if you still use Discloud)
- `auto-discloud-upload.ps1` (optional Discloud helper)

## 1) Prepare local env
1. Copy `.env.example` to `.env`
2. Fill all required values in `.env`

## 2) Push to GitHub
```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 3) Deploy on Render (Background Worker)
- Create new **Background Worker**
- Connect this GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `python Witan.py`
- Add all env vars from `.env` to Render Dashboard

`render.yaml` is already included if you want Infra-as-Code setup.
