import time
import re
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-US")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )

    # Uncomment if you want it hidden
    # options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    return driver


def close_popup_if_present(driver):
    popup_selectors = [
        (By.XPATH, "//button[contains(text(),'✕')]"),
        (By.XPATH, "//button[contains(text(),'X')]"),
        (By.CSS_SELECTOR, "button._2KpZ6l._2doB4z"),
    ]

    for by, selector in popup_selectors:
        try:
            btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((by, selector))
            )
            btn.click()
            print("Closed popup")
            time.sleep(2)
            return
        except:
            pass


def scroll_page(driver):
    scroll_points = [400, 1000, 1800, 2600, 3400]
    for point in scroll_points:
        try:
            driver.execute_script(f"window.scrollTo(0, {point});")
            time.sleep(1.5)
        except:
            pass


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_reviews_from_soup(soup):
    reviews = []

    # Find all "Review for:" blocks
    review_for_divs = soup.find_all(
        lambda tag: tag.name == "div"
        and tag.get_text(strip=True).startswith("Review for:")
    )

    print(f"Found {len(review_for_divs)} 'Review for:' blocks")

    for review_for_div in review_for_divs:
        try:
            # Title is usually the previous sibling div
            title_div = review_for_div.find_previous_sibling("div")
            body_div = review_for_div.find_next_sibling("div")

            title = ""
            body = ""

            if title_div:
                title = clean_text(title_div.get_text(" ", strip=True))

            if body_div:
                body = clean_text(body_div.get_text(" ", strip=True))
                body = body.replace("READ MORE", "").strip()

            full_review = f"{title}. {body}".strip(". ").strip()

            if full_review and len(full_review.split()) >= 4:
                reviews.append(full_review)

        except Exception:
            pass

    # Fallback: extract bodies only if needed
    if not reviews:
        body_spans = soup.find_all("span", class_="css-1jxf684")
        print(f"Fallback body spans found: {len(body_spans)}")

        for span in body_spans:
            text = clean_text(span.get_text(" ", strip=True))
            if text and len(text.split()) >= 2:
                reviews.append(text)

    # Deduplicate
    final_reviews = []
    seen = set()

    for review in reviews:
        review = review.strip()
        if review and review not in seen:
            seen.add(review)
            final_reviews.append(review)

    return final_reviews


def scrape_flipkart_reviews(base_url, max_pages=10):
    driver = setup_driver()
    all_reviews = []

    try:
        for page in range(1, max_pages + 1):
            url = f"{base_url}&page={page}"
            print(f"\nScraping page {page}: {url}")

            driver.get(url)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            time.sleep(5)
            close_popup_if_present(driver)
            scroll_page(driver)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            page_reviews = extract_reviews_from_soup(soup)

            print(f"Found {len(page_reviews)} reviews on page {page}")

            if page_reviews:
                print("Sample review:", page_reviews[0][:200])
            else:
                print("No reviews found on this page")
                print("Page title:", driver.title)
                print("Current URL:", driver.current_url)

                with open(f"debug_page_{page}.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"Saved debug_page_{page}.html")

            all_reviews.extend(page_reviews)
            time.sleep(2)

    except Exception as e:
        print("Error occurred:", e)

    finally:
        driver.quit()

    # Final deduplication
    final_reviews = []
    seen = set()

    for review in all_reviews:
        review = review.strip()
        if review and review not in seen:
            seen.add(review)
            final_reviews.append(review)

    df = pd.DataFrame({"review_text": final_reviews})
    return df


if __name__ == "__main__":
    base_url = "https://www.flipkart.com/apple-iphone-16-black-128-gb/product-reviews/itmb07d67f995271?pid=MOBH4DQFG8NKFRDY&lid=LSTMOBH4DQFG8NKFRDYNBDOZI&marketplace=FLIPKART"

    # test with 1 page first
    df_reviews = scrape_flipkart_reviews(base_url, max_pages=12)

    print("\nTotal reviews collected:", len(df_reviews))
    print(df_reviews.head(10))

    df_reviews.to_csv("flipkart_reviews.csv", index=False, encoding="utf-8")
    print("Saved to flipkart_reviews.csv")