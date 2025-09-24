# scrape_tengrinews_threads.py
import re, time, json, random, pathlib
from datetime import datetime
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from dateutil import parser as dateparser
except ImportError:
    raise ImportError("Установи dateutil: pip install python-dateutil")

BASE = "https://tengrinews.kz"
LISTING_FMT = BASE + "/news/page/{page}/"
OUT_DIR = pathlib.Path("data_tengrinews")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = OUT_DIR / "tengrinews_ru.jsonl"
SEEN_PATH = OUT_DIR / "seen_urls.txt"

# --------- сессия с ретраями ----------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; TengriParser/1.0; +https://example.org)",
    "Accept-Language": "ru-RU,ru;q=0.9"
})
ADAPT = requests.adapters.HTTPAdapter(max_retries=3)
SESSION.mount("http://", ADAPT)
SESSION.mount("https://", ADAPT)

# --------- хелперы ----------
ART_URL_RE = re.compile(r"^https?://tengrinews\.kz/[^/]+/.+-(\d+)/?$")

def load_seen():
    if SEEN_PATH.exists():
        return set(SEEN_PATH.read_text(encoding="utf-8").splitlines())
    return set()

def save_seen(seen):
    SEEN_PATH.write_text("\n".join(sorted(seen)), encoding="utf-8")

def write_jsonl(record):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_article_links(listing_html):
    soup = BeautifulSoup(listing_html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(BASE, href)
        if "tengrinews.kz" in href and ART_URL_RE.match(href):
            links.add(href.split("#")[0])
    return sorted(links)

def parse_article(html, url):
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    published = None
    for prop in ("article:published_time", "og:article:published_time", "og:updated_time"):
        m = soup.find("meta", attrs={"property": prop})
        if m and m.get("content"):
            published = m["content"]; break
    if not published:
        t = soup.find("time")
        if t and (t.get("datetime") or t.text):
            published = t.get("datetime") or t.get_text(strip=True)
    if not published:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "datePublished" in data:
                    published = data["datePublished"]; break
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict) and "datePublished" in d:
                            published = d["datePublished"]; break
            except Exception:
                pass

    published_dt = None
    if published:
        try:
            published_dt = dateparser.parse(published)
        except Exception:
            published_dt = None

    body_texts = []
    candidates = []
    for attr, val in [("itemprop","articleBody"), ("class", re.compile("content|article|tn-text|tn-article"))]:
        found = soup.find_all(True, attrs={attr: val})
        candidates.extend(found)
    if not candidates:
        candidates = [soup]

    seen_p = set()
    for node in candidates:
        for p in node.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt and txt not in seen_p:
                seen_p.add(txt)
                body_texts.append(txt)

    junk_starts = ("Читайте также", "TENGRI", "Фото:", "Видео:", "Поделиться", "ПОСЛЕДНИЕ НОВОСТИ")
    body = "\n".join([t for t in body_texts if not any(t.startswith(js) for js in junk_starts)])
    if body_texts[:1] and body_texts.count(body_texts[0]) > 1:
        body = "\n".join([t for i,t in enumerate(body_texts) if t != body_texts[0] or i == 0])

    author = None
    author_link = soup.find("a", href=re.compile(r"/author/"))
    if author_link:
        author = author_link.get_text(strip=True)

    tags = [a.get_text(strip=True) for a in soup.select("a[href*='/tag/']")] or None

    return {
        "url": url,
        "title": title,
        "published": published,
        "published_dt": published_dt.isoformat() if published_dt else None,
        "author": author,
        "tags": tags,
        "text": body
    }

def fetch_and_parse(url):
    try:
        rr = SESSION.get(url, timeout=25)
        if rr.status_code != 200:
            return None
        art = parse_article(rr.text, url)
        if not art.get("published_dt"):
            return None
        year = int(art["published_dt"][:4])
        if year < 2024 or year > 2025:
            return None
        if art.get("title") and art.get("text"):
            return art
    except Exception:
        return None
    return None

# --------- основной цикл ----------
def crawl_listings(start_page=1, end_page=None, max_articles=None, workers=10):
    seen = load_seen()
    total_saved = 0
    page = start_page

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            if end_page and page > end_page: break
            listing_url = LISTING_FMT.format(page=page)
            try:
                r = SESSION.get(listing_url, timeout=20)
            except Exception:
                page += 1; continue

            if r.status_code == 404:
                break
            if r.status_code != 200:
                page += 1; continue

            links = extract_article_links(r.text)
            if not links:
                break

            new_links = [u for u in links if u not in seen]
            futures = [executor.submit(fetch_and_parse, u) for u in new_links]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Page {page}"):
                art = fut.result()
                if art:
                    write_jsonl(art)
                    seen.add(art["url"])
                    total_saved += 1

                if max_articles and total_saved >= max_articles:
                    save_seen(seen)
                    return total_saved

            save_seen(seen)
            page += 1

    return total_saved

if __name__ == "__main__":
    saved = crawl_listings(start_page=1, max_articles=50000, workers=10)
    print(f"Saved {saved} articles (2024–2025) to {JSONL_PATH}")
