import sys
from playwright.sync_api import sync_playwright

mode = sys.argv[1]
url  = sys.argv[2]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")

    if mode == "listing":
        page.wait_for_selector("a[href*='/job/']", timeout=15000)
        prev_count = 0
        for _ in range(10):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)
            count = page.locator("a[href*='/job/']").count()
            if count == prev_count:
                break
            prev_count = count
        print(page.content())

    elif mode == "job":
        job_id = url.rstrip("/").split("/")[-1]
        page.wait_for_selector(f'[x-ref="job-post-body-{job_id}"]', timeout=10000)
        print(page.inner_html(f'[x-ref="job-post-body-{job_id}"]'))

    browser.close()
