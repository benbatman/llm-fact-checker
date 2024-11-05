import asyncio
import aiohttp
from bs4 import BeautifulSoup


async def fetch_with_retries(session, url, retries=3, backoff_factor=2):
    attempt = 0
    while attempt < retries:
        try:
            async with session.get(url) as response:
                return await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            attempt += 1
            wait_time = backoff_factor**attempt
            print(f"Retrying {url} in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    return None  # return none if all retries fail


async def get_video_links(url: str, session: aiohttp.ClientSession) -> list[str]:
    try:
        url_content = await fetch_with_retries(session, url)
        if not url_content:
            print(f"Failed to fetch {url}")
            return []  # return empty list on failure

        # mp3 link is in audio html tag as source in div audioplayer
        soup = BeautifulSoup(url_content, "html.parser")
        # Need ul tag with class playlist
        playlist = soup.find("ul", class_="playlist")
        # All a tags in playlist are links to other short episodes
        if playlist:
            print("Found playlist")
            show_links: list[str] = [
                link.get("href") for link in playlist.find_all("a")
            ]
            return show_links

    except aiohttp.ClientError as e:
        print(f"Error: {e}")
        return []


async def main(urls):
    timeout = aiohttp.ClientTimeout(total=60)
    # urls_to_try = urls[:30]
    # print(urls_to_try)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        semaphore = asyncio.Semaphore(5)

        async def limited_get_video_links(link: str):
            async with semaphore:
                return await get_video_links(link, session)

        tasks = [limited_get_video_links(link) for link in urls]
        show_links = await asyncio.gather(*tasks)
        # print(mp3s_and_show_links)
        print("number of show links found: ", len(show_links))
        return show_links


if __name__ == "__main__":

    with open("all_links.txt", "r") as f:
        urls = f.read().splitlines()

    show_links = asyncio.run(main(urls))
    # Remove None values from list
    show_links = [link for link in show_links if link]

    # Write show links to file
    with open("show_links.txt", "w") as f:
        for show_link in show_links:
            f.write(",".join(show_link) + "\n")

    print("initial number of total urls: ", len(urls))
    # Add show_links to urls and ensure no duplicates
    urls.extend([link.strip() for show_link in show_links for link in show_link])
    urls = list(set(urls))
    print("new number of urls: ", len(urls))
    print("Done!")
    # Write to new list
    with open("all_links_final.txt", "w") as f:
        for url in urls:
            f.write(url + "\n")
