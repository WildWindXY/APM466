import csv
from datetime import datetime

import requests
from playwright.sync_api import sync_playwright

start_date = "20250106"
end_date = "20250120"
start_dt = datetime.strptime(start_date, "%Y%m%d")
end_dt = datetime.strptime(end_date, "%Y%m%d")


def get_bond_links(browser):
    urls = [
        "https://markets.businessinsider.com/bonds/finder?p=1&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19",
        "https://markets.businessinsider.com/bonds/finder?p=2&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19",
        "https://markets.businessinsider.com/bonds/finder?p=1&borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
    ]
    all_bonds = []
    total_urls = len(urls)
    for i, url in enumerate(urls, start=1):
        print(f"Processing bond finder URL {i}/{total_urls}")
        page = browser.new_page()
        page.goto(url, timeout=60000)
        rows = page.query_selector_all("table tbody tr")
        for row in rows:
            cols = row.query_selector_all("td")
            if len(cols) >= 7:
                a = cols[0].query_selector("a")
                if a:
                    href = a.get_attribute("href")
                    if href:
                        all_bonds.append(f"https://markets.businessinsider.com{href}")
        page.close()
    return all_bonds


def process_bond(browser, bond_url):
    chart_url = {"url": None}

    def log_request(request):
        if "Chart_GetChartData" in request.url and chart_url["url"] is None:
            chart_url["url"] = request.url.replace("19700201", "20241130")

    page = browser.new_page()
    page.on("request", log_request)
    page.goto(bond_url, timeout=60000)
    rows = page.query_selector_all("tbody.table__tbody tr")
    bond_data = {}
    for row in rows:
        cols = row.query_selector_all("td")
        if len(cols) == 2:
            key = cols[0].inner_text().strip().lower()
            value = cols[1].inner_text().strip()
            bond_data[key] = value
    keys_to_keep = {"coupon", "isin", "issue date", "maturity date"}
    clean_bond_data = {k: v for k, v in bond_data.items() if k in keys_to_keep}
    info_data = {
        "Coupon": clean_bond_data.get("coupon", ""),
        "ISIN": clean_bond_data.get("isin", ""),
        "Issue Date": clean_bond_data.get("issue date", ""),
        "Maturity Date": clean_bond_data.get("maturity date", "")
    }
    aggregated_data = {}
    allowed_days = [6, 7, 8, 9, 10, 13, 14, 15, 16, 17]
    if chart_url["url"]:
        response = requests.get(chart_url["url"])
        chart_data = response.json()
        for record in chart_data:
            try:
                dt = datetime.strptime(record["Date"], "%Y-%m-%d %H:%M")
            except Exception:
                continue
            if start_dt <= dt <= end_dt and dt.day in allowed_days:
                key = f"{dt.day}"
                if key not in aggregated_data:
                    aggregated_data[key] = record["Close"]
        for day in allowed_days:
            key = f"{day}"
            if key not in aggregated_data:
                aggregated_data[key] = ""
    else:
        for day in allowed_days:
            aggregated_data[f"{day}"] = ""
    final_data = {**info_data, **aggregated_data}
    page.close()
    return final_data


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    bond_links = get_bond_links(browser)
    print(f"{len(bond_links)} links found!")
    if not bond_links:
        print("No bond links found.")
        browser.close()
        exit(1)

    all_rows = []
    total_bonds = len(bond_links)
    for i, bond_url in enumerate(bond_links, start=1):
        print(f"Processing bond {i}/{total_bonds}: {bond_url}")
        row_data = process_bond(browser, bond_url)
        all_rows.append(row_data)
    fieldnames = [
        "Coupon", "ISIN", "Issue Date", "Maturity Date",
        "6", "7", "8", "9", "10",
        "13", "14", "15", "16", "17"
    ]
    with open("bond_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    browser.close()
