import asyncio
import xml.etree.ElementTree as ET
import aiohttp
import os
import re
import json
import time


from bs4 import BeautifulSoup
import requests
from tqdm.asyncio import tqdm_asyncio

from utils import get_sub_videos


async def fetch_sitemap(session: aiohttp.ClientSession, url):
    async with session.get(url) as response:
        if response.status == 200:
            print(f"Successfully fetched {url}")
            return await response.text()
        else:
            return None


def parse_sitemap(content):
    if content:
        tree = ET.ElementTree(ET.fromstring(content))
        root = tree.getroot()
        print(f"Number of sitemap tags are {len(root)}")
        urls = []

        namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check if it's an index sitemap
        if root.tag.endswith("sitemapindex"):
            print("index")
            for sitemap in root.findall("sitemap:sitemap", namespaces=namespace):
                loc = sitemap.find("sitemap:loc", namespaces=namespace).text
                if loc:
                    urls.append(loc)
        elif root.tag.endswith("urlset"):
            for url in root.findall("sitemap:url", namespaces=namespace):
                loc = url.find("sitemap:loc", namespaces=namespace).text
                urls.append(loc)

        return urls
    else:
        print("Failed to fetch sitemap")
        return []


async def scrape_sitemap_index(url):
    async with aiohttp.ClientSession() as session:
        total_urls = []

        # get the main sitemap
        main_sitemap_content = await fetch_sitemap(session, url)
        main_sitemaps = parse_sitemap(main_sitemap_content)

        # Fetch all individual sitemaps concurrently
        tasks = [fetch_sitemap(session, sitemap_url) for sitemap_url in main_sitemaps]
        sitemap_content = await asyncio.gather(*tasks)
        print(sitemap_content)

        # Parse all the individual sitemaps and extact URLs
        # This should be all urls from all sitemaps
        for content in sitemap_content:
            total_urls.extend(parse_sitemap(content))

        return total_urls


# Function to get url content and parse into fields
async def get_url_content(url):
    """
    Take in url and return the parsed content
    """
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Error fetching {url}: {response.status}")
                return None

            url_contents = await response.text()
            soup = BeautifulSoup(url_contents, "html.parser")
            # article is in div == post__col-right
            # title is in h1, post__title
            # timestamp is in time, post__date
            # Text is in <article> --> <div class='body-text> in <p> tags
            title = soup.find("h1", class_="post__title")
            if title:
                title = title.text.strip()
            has_transcript = soup.find("ul", class_="video-transcript")

            # Get article content
            if title:
                timestamp = soup.find("time", class_="post__date")
                if timestamp:
                    timestamp = timestamp.text.strip()
                text = soup.find("div", class_="body-text")
                # Possiblity this url has a video and no transcript, check if text variable has video tag
                if text.find("video"):
                    # Url has video but no transcript
                    pass

                body_text = ""
                if text:
                    for p in text.find_all("p"):
                        body_text += p.text.strip() + "\n"

                article_content = {
                    "title": title,
                    "timestamp": timestamp,
                    "body_text": body_text,
                    "url": url,
                }
                return article_content

            # Get transcript
            elif has_transcript:
                title = soup.find("h1", class_="video-single__title")
                if title:
                    title = title.text.strip()
                post_date = soup.find("time", class_="video-single__date")
                if post_date:
                    post_date = post_date.text.strip()
                # Get all p tags in transcript
                transcript = ""
                for p in has_transcript.find_all("p"):
                    transcript += p.text.strip() + "\n"

                video_content = {
                    "title": title,
                    "post_date": post_date,
                    "transcript": transcript,
                    "url": url,
                }
                return video_content


async def get_snopes_url_content(url):
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Error fetching {url}: {response.status}")
                    return None

                response = await session.get(url)
                url_contents = await response.text()
                soup = BeautifulSoup(url_contents, "html.parser")
                article_content = soup.find("article", id="article-content")
                title_container = soup.find("section", class_="title-container")
                if article_content:
                    claim_cont = article_content.find("div", class_="claim_cont")
                    if claim_cont:
                        claim_cont = claim_cont.text.strip()

                    claim_rating = article_content.find(
                        "div", class_="rating_title_wrap"
                    )
                    if claim_rating:
                        claim_rating = claim_rating.text.strip()
                        claim_rating = re.sub(r"[\n\t]", "", claim_rating).replace(
                            "About this rating", ""
                        )

                    published_date = title_container.find("h3", class_="publish_date")
                    if published_date:
                        published_date = published_date.text.replace(
                            "Published ", ""
                        ).strip()

                    context = article_content.find(
                        "div", class_="outer_fact_check_context"
                    )
                    # find p tags in context
                    context_text = ""
                    article_text = ""
                    if context:
                        for p in context.find_all("p"):
                            context_text += p.text.strip() + "\n"

                        for p in article_content.find_all(
                            "p",
                        ):
                            article_text += p.text.strip() + "\n"
                        article_text = re.sub(r"[\n\t]", "", article_text).replace(
                            "About this rating", ""
                        )

                    snopes_content = {
                        "claim_cont": claim_cont,
                        "claim_rating": claim_rating,
                        "context_text": context_text,
                        "article_text": article_text,
                        "published_date": published_date,
                        "url": url,
                    }

                    return snopes_content
        except aiohttp.ClientError as e:
            print(f"Error: {e}")
            return None


def scrape_all_sitemaps(url):
    return asyncio.run(scrape_sitemap_index(url))


semaphore = asyncio.Semaphore(3)


async def main():
    main_sitemap_url = "https://www.pbs.org/newshour/sitemaps/sitemap.xml"
    if os.path.exists("uncompleted_links.txt"):
        with open("uncompleted_links.txt", "r") as f:
            pbs_links = f.read().splitlines()
    elif os.path.exists("all_links_final.txt"):
        with open("all_links_final.txt", "r") as f:
            pbs_links = f.read().splitlines()
    else:
        pbs_links = scrape_all_sitemaps(main_sitemap_url)

    print("total links found: ", len(pbs_links))

    scrape_pbs_urls = True
    print("Scraping PBS urls...")
    if scrape_pbs_urls:

        async def limited_get_url_content(url, timeout_urls: list):
            async with semaphore:
                try:
                    return await get_url_content(url)
                except asyncio.TimeoutError:
                    print(f"Timeout error for URL: {url}")
                    timeout_urls.append(url)
                    return None
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                    timeout_urls.append(url)
                    return None

        batch_size = 10
        sleep_duration = 10
        results = []
        timeout_urls = []
        print("Processing URLs...")
        for i in tqdm_asyncio(range(0, len(pbs_links), batch_size)):
            batch = pbs_links[i : i + batch_size]

            tasks = [limited_get_url_content(url, timeout_urls) for url in batch]

            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            print(
                f"Processed {i + batch_size} URLs, sleeping for {sleep_duration} seconds..."
            )
            if i % 50 == 0 and i != 0:
                with open("extra_pbs_results.json", "w") as f:
                    json.dump(results, f, indent=4)

        return timeout_urls

    get_snopes_urls = False
    if get_snopes_urls:
        snopes_sitemap_index_url = "https://www.snopes.com/sitemaps/sitemap-index.xml"
        # Get all snopes urls
        snopes_links = await scrape_sitemap_index(snopes_sitemap_index_url)
        print("total snopes links found: ", len(snopes_links))
        with open("snopes_links.txt", "w") as f:
            for link in snopes_links:
                f.write(link + "\n")

    scrape_snopes_urls = False
    if scrape_snopes_urls:
        with open("snopes_links_1.txt", "r") as f:
            snopes_links = f.read().splitlines()

        # Read in json
        with open("snopes_results_0.json", "r") as f:
            results = json.load(f)
            print("Number of results rounded: ", round(len(results) / 50) * 50)
        # Just use half of the links
        snopes_links = snopes_links[round(len(results) / 50) * 50 : len(snopes_links)]

        async def limited_get_snopes_url_content(url, timeout_urls: list):
            async with semaphore:
                try:
                    return await get_snopes_url_content(url)
                except:
                    print(f"Timeout error for URL: {url}")
                    timeout_urls.append(url)
                    return None

        batch_size = 10
        sleep_duration = 5
        results = []
        timeout_urls = []
        print("Processing URLs...")
        for i in tqdm_asyncio(range(0, len(snopes_links), batch_size)):
            batch = snopes_links[i : i + batch_size]

            tasks = [limited_get_snopes_url_content(url, timeout_urls) for url in batch]

            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            print(
                f"Processed {i + batch_size} URLs, sleeping for {sleep_duration} seconds..."
            )
            if i % 50 == 0 and i != 0:
                # Save results to a JSON file
                # with open("snopes_results.json", "r") as f:
                #     print("Saving results to JSON file...")
                #     try:
                #         existing_data = json.load(f)
                #     except json.decoder.JSONDecodeError:
                #         existing_data = []

                # existing_data.extend(
                #     [result for result in results if result is not None]
                # )

                with open("snopes_results_1.json", "w") as f:
                    json.dump(results, f, indent=4)

            await asyncio.sleep(sleep_duration)

        return timeout_urls


if __name__ == "__main__":
    timeout_urls = asyncio.run(main())
    # Write to txt file
    with open("timeout_urls_pbs.txt", "w") as f:
        for url in timeout_urls:
            f.write(url + "\n")
