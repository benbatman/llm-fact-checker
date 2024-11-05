import bs4 as BeautifulSoup


def get_sub_videos(playlist) -> list[str]:
    # All a tags in playlist are links to other short episodes
    if playlist:
        show_links: list[str] = [link.get("href") for link in playlist.find_all("a")]
        return show_links


def get_transcripts(show_links: list[str]):
    pass
