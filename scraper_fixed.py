import subprocess
import re
import json
import yaml
import os
from bs4 import BeautifulSoup
from datetime import date

OUTPUT_DIR   = "ai_jobs_output"
BASE_LISTING = "https://builtin.com/jobs?search=ai%20engineer&country=USA&allLocations=true"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_pw(mode, url, timeout=60):
    r = subprocess.run(
        ["python", "pw_scraper.py", mode, url],
        capture_output=True, text=True, timeout=timeout
    )
    if r.returncode != 0:
        raise Exception(r.stderr[:300])
    return r.stdout

def get_job_urls(listing_html):
    soup = BeautifulSoup(listing_html, "lxml")
    urls = set()
    for a in soup.find_all("a", href=re.compile(r"/job/")):
        href = a["href"]
        if href.startswith("/job/"):
            href = "https://builtin.com" + href
        parts = href.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1].isdigit():
            urls.add(href)
    return urls

def get_section(body, header_text):
    header_aliases = {
        "responsibilities": [
            "responsibilities", "about the role", "what you'll do", "the role",
            "your role", "position overview", "job description", "about this role",
            "role overview", "what you will do", "what you'll be doing",
            "your responsibilities", "key responsibilities", "core responsibilities",
            "primary responsibilities", "main responsibilities", "day to day",
            "day-to-day", "in this role", "about the job", "the opportunity",
            "what we need", "the position", "job summary", "role summary",
            "position summary", "about this position", "what you'll own",
            "what you will own", "your impact", "the work", "your work",
            "what you'll build", "what you will build", "what you'll drive",
            "mission", "your mission", "the mission", "scope of work",
            "job duties", "duties", "duties and responsibilities",
            "essential duties", "essential functions", "essential responsibilities",
            "key duties", "role description", "job overview", "overview",
            "position description", "what does the job involve", "you will",
            "as an", "as a", "in this position", "your day",
        ],
        "basic qualifications": [
            "basic qualifications", "required qualifications", "minimum qualifications","Experience & Portfolio",
            "requirements", "what you need", "you have", "what we're looking for",
            "what you bring", "who you are", "must have", "must-have",
            "required skills", "required experience", "required background",
            "what you'll need", "what you will need", "qualifications",
            "minimum requirements", "basic requirements", "core requirements",
            "skills required", "experience required", "you must have",
            "you should have", "you need", "we require", "we need",
            "mandatory", "essential skills", "essential qualifications",
            "key qualifications", "key requirements", "key skills",
            "technical requirements", "technical qualifications",
            "about you", "who we're looking for", "who we are looking for",
            "ideal candidate", "the ideal candidate", "candidate requirements",
            "candidate profile", "your profile", "your background",
            "your experience", "your skills", "your qualifications",
            "what makes you a fit", "what we expect", "expectations",
            "you qualify if", "to be successful", "to succeed in this role",
        ],
        "preferred qualifications": [
            "preferred qualifications", "nice to have", "bonus points", "preferred",
            "plus if you have", "nice-to-have", "bonus if you have",
            "preferred skills", "preferred experience", "preferred background",
            "additional qualifications", "additional skills", "additional requirements",
            "it's a plus", "it is a plus", "a plus", "great if you have",
            "ideally you have", "ideally you will have", "ideally",
            "would be great", "would be a bonus", "would be nice",
            "extra points", "extra credit", "stand out if",
            "you'll stand out", "you will stand out", "sets you apart",
            "what sets you apart", "desirable", "desirable skills",
            "desirable qualifications", "advantageous", "not required but",
            "optional but preferred", "we'd love if", "we would love if",
            "get bonus points", "brownie points",
        ],
    }

    aliases = header_aliases.get(header_text.lower(), [header_text.lower()])

    for tag in body.find_all(["h2", "h3", "h4", "strong", "b"]):
        if any(alias in tag.get_text().lower() for alias in aliases):
            items = []
            for sibling in tag.find_next_siblings():
                if sibling.name in ["h2", "h3", "h4"] or (
                    sibling.name in ["strong", "b"] and len(sibling.get_text()) < 60
                ):
                    break
                for el in sibling.find_all(["span", "li", "p"]):
                    text = el.get_text(separator=" ", strip=True)
                    if text and len(text) > 20:
                        items.append(text)
                if not sibling.find_all(["span", "li", "p"]):
                    text = sibling.get_text(separator=" ", strip=True)
                    if text and len(text) > 20:
                        items.append(text)
            if items:
                return items

    full = body.get_text(separator="\n", strip=True)
    end_markers = [
        "basic qualifications", "required qualifications", "minimum qualifications",
        "preferred qualifications", "nice to have", "benefits", "equal opportunity",
        "background verification", "you will have access", "please be aware",
        "compensation", "salary", "what we offer", "we offer", "perks",
        "about us", "about the company", "our company", "who we are",
        "apply now", "how to apply",
    ]
    capturing, items = False, []
    for line in full.split("\n"):
        line_lower = line.strip().lower()
        if any(alias in line_lower for alias in aliases):
            capturing = True
            continue
        if capturing:
            if any(marker in line_lower for marker in end_markers):
                break
            if line.strip() and len(line.strip()) > 20:
                items.append(line.strip())
    return items if items else None

def get_travel(body):
    full = body.get_text(separator=" ", strip=True)
    patterns = [
        r"travel(?:ing)? (?:is )?up to (\d+)\s*%",
        r"up to (\d+)\s*% travel",
        r"(\d+)\s*% travel(?:ing)?",
        r"travel(?:ing)? (?:up to )?(\d+)\s*%",
        r"requires? (\d+)\s*% travel",
        r"travel requirement[s]?[^\d]*(\d+)\s*%",
        r"travel[^\d]*(\d+)\s*%",
    ]
    for p in patterns:
        m = re.search(p, full, re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def get_experience(body):
    full = body.get_text(separator=" ", strip=True)
    patterns = [
        r"(\d+)\+?\s*years? of experience",
        r"(\d+)\+?\s*years? experience",
        r"minimum (?:of )?(\d+)\+?\s*years?",
        r"at least (\d+)\+?\s*years?",
        r"(\d+)\+?\s*years? (?:of )?(?:professional|relevant|related|industry|work|hands.on)",
        r"experience[^\d]*(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years? in",
        r"(\d+)\s*-\s*\d+\s*years?",
    ]
    for p in patterns:
        m = re.search(p, full, re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def scrape_job(url):
    job_id = url.rstrip("/").split("/")[-1]
    slug   = url.rstrip("/").split("/")[-2]
    html   = run_pw("job", url)
    body   = BeautifulSoup(html, "lxml")
    return {
        "job_id":                   job_id,
        "job_title":                slug.replace("-", " ").title(),
        "source_url":               url,
        "date_scraped":             str(date.today()),
        "responsibilities":         get_section(body, "responsibilities"),
        "basic_qualifications":     get_section(body, "basic qualifications"),
        "preferred_qualifications": get_section(body, "preferred qualifications"),
        "travel_percent":           get_travel(body),
        "experience_years_min":     get_experience(body),
    }

def save_yaml(data):
    fname = f"{data['job_id']}_{data['job_title'].replace(' ', '_')}.yaml"
    path  = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return path

# --- step 1: collect all job URLs dynamically ---
all_job_urls = set()
page_num = 1
consecutive_empty = 0

while True:
    listing_url = BASE_LISTING if page_num == 1 else f"{BASE_LISTING}&page={page_num}"
    print(f"Listing page {page_num}: {listing_url}")
    try:
        html = run_pw("listing", listing_url)
        urls = get_job_urls(html)
        new  = urls - all_job_urls
        all_job_urls.update(new)
        print(f"  found {len(urls)} jobs | total so far: {len(all_job_urls)}")
        if not urls:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("  3 consecutive empty pages — stopping pagination")
                break
        else:
            consecutive_empty = 0
    except Exception as e:
        print(f"  failed: {e}")
        consecutive_empty += 1
        if consecutive_empty >= 3:
            print("  3 consecutive failures — stopping pagination")
            break
    page_num += 1

print(f"\nTotal jobs to scrape: {len(all_job_urls)}")

# --- step 2: scrape each job ---
for i, url in enumerate(sorted(all_job_urls), 1):
    job_id = url.rstrip("/").split("/")[-1]
    print(f"[{i}/{len(all_job_urls)}] Scraping {job_id}")
    try:
        data = scrape_job(url)
        path = save_yaml(data)
        print(f"  saved → {path}")
    except Exception as e:
        print(f"  failed: {e}")

print(f"\nDone. {len(all_job_urls)} jobs → ./{OUTPUT_DIR}/")
