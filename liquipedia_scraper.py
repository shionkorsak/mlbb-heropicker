"""
Liquipedia MLBB Web Scraper
Uses the Liquipedia API to scrape hero statistics from tournament pages.
API Documentation: https://liquipedia.net/api-terms-of-use
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
import os
from typing import Dict, List, Optional, Tuple


class LiquipediaScraper:
    """Scrape MLBB statistics from Liquipedia using their API."""
    
    API_URL = "https://liquipedia.net/mobilelegends/api.php"
    
    NON_HEROES = {
        'team liquid', 'rrq', 'evos', 'onic', 'geek fam', 'echo', 'blacklist',
        'fnatic', 'bren', 'nxpe', 'omega', 'smart omega', 'tlph', 'nexplay',
        'indonesia', 'philippines', 'malaysia', 'cambodia', 'singapore', 'vietnam',
        'thailand', 'brazil', 'latam', 'turkey', 'russia', 'mena', 'japan',
        'team liquid id', 'rrq hoshi', 'evos legends', 'alter ego', 'aura fire',
        'bigetron', 'aerowolf', 'rex regum', 'falcon', 'todak', 'suhaz',
        'homebois', 'orange', 'team flash', 'burmese ghouls', 'see you soon',
        'ap bren', 'rsg', 'tnc', 'aurora', 'vamos', 'pain gaming', 'loud',
        'team secret', 'moonton', 'mpl', 'show', 'details', 'hide', 'statistics',
        'uzbekistan', 'egypt', 'iran', 'nepal', 'mongolia', 'argentina', 'colombia',
        'south africa', 'peru', 'mexico', 'chile', 'ecuador', 'bolivia', 'paraguay',
        'china', 'taiwan', 'hong kong', 'korea', 'india', 'pakistan', 'bangladesh',
        'myanmar', 'laos', 'brunei', 'timor-leste', 'australia', 'new zealand',
        'united states', 'canada', 'uk', 'germany', 'france', 'spain', 'italy',
        'portugal', 'netherlands', 'belgium', 'poland', 'ukraine', 'sweden',
        'norway', 'denmark', 'finland', 'czech republic', 'austria', 'switzerland',
        'greece', 'romania', 'hungary', 'bulgaria', 'croatia', 'serbia', 'morocco',
        'algeria', 'tunisia', 'nigeria', 'kenya', 'ghana', 'south korea', 'north korea',
        'cam', 'iri', 'idn', 'mas', 'uzb', 'egy', 'npl', 'mgl', 'arg', 'col', 'zaf',
        'dewa', 'geek', 'dewa united esports', 'geek fam id', 'onic esports',
    }
    
    def __init__(self, delay: float = 2.0):
        """Initialize scraper with request delay to respect rate limits."""
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MLBBHeroPicker/1.0 (https://github.com/mlbb-heropicker; Educational Project)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
        })
        self.last_request_time = 0
    
    def _respectful_request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Make a request with delay to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response
            else:
                print(f"  Error: HTTP {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"  Error: {e}")
            return None
    
    def get_page_html(self, page_title: str) -> Optional[str]:
        """Get the HTML content of a Liquipedia page using the API."""
        params = {
            'action': 'parse',
            'page': page_title,
            'format': 'json',
            'prop': 'text',
        }
        
        response = self._respectful_request(self.API_URL, params)
        if not response:
            return None
        
        try:
            data = response.json()
            if 'parse' in data and 'text' in data['parse']:
                return data['parse']['text']['*']
            elif 'error' in data:
                print(f"  API Error: {data['error'].get('info', 'Unknown error')}")
                return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error: {e}")
            return None
        
        return None
    
    def url_to_page_title(self, url: str) -> str:
        """Convert a Liquipedia URL to a page title for the API."""
        path = url.replace('https://liquipedia.net/mobilelegends/', '')
        path = path.rstrip('/')
        path = path.replace(' ', '_')
        return path
    
    def scrape_hero_statistics(self, url_or_title: str) -> Optional[pd.DataFrame]:
        """Scrape hero statistics from a tournament statistics page."""
        if url_or_title.startswith('http'):
            page_title = self.url_to_page_title(url_or_title)
        else:
            page_title = url_or_title
        
        if not page_title.endswith('Statistics') and not page_title.endswith('statistics'):
            page_title = page_title + '/Statistics'
        
        print(f"Fetching: {page_title}")
        
        html = self.get_page_html(page_title)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        hero_stats = []
        seen_heroes = set()
        
        tables = soup.find_all('table', class_='wikitable')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 3:
                continue
            
            all_headers = []
            for header_row in rows[:2]:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
                if headers:
                    all_headers.extend(headers)
            
            header_text = ' '.join(all_headers).lower()
            
            if 'hero' not in header_text:
                continue
            
            is_main_stats = 'picks' in header_text or 'bans' in header_text or 'blue' in header_text
            
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 4:
                    continue
                
                hero_name = None
                hero_cell_idx = 0
                
                for idx, cell in enumerate(cells[:4]):
                    if cell.get('class') and 'stats-table-number' in ' '.join(cell.get('class', [])):
                        continue
                    
                    link = cell.find('a')
                    if link:
                        title = link.get('title', '')
                        text = link.get_text(strip=True)
                        if title and not title.startswith('File:') and not title.startswith('Category:'):
                            hero_name = title
                            hero_cell_idx = idx
                            break
                        elif text and len(text) > 1 and not text.isdigit():
                            hero_name = text
                            hero_cell_idx = idx
                            break
                
                if not hero_name or hero_name in seen_heroes:
                    continue
                
                if hero_name.lower() in self.NON_HEROES:
                    continue
                if len(hero_name) < 2 or hero_name.isdigit():
                    continue
                
                seen_heroes.add(hero_name)
                
                numbers = []
                percentages = []
                
                for cell in cells[hero_cell_idx + 1:]:
                    text = cell.get_text(strip=True)
                    if '%' in text:
                        percentages.append(self._parse_percent(text))
                    elif text.isdigit() or (text.replace('.', '').isdigit() and '.' in text):
                        try:
                            numbers.append(int(float(text)))
                        except ValueError:
                            pass
                
                total_games = numbers[0] if len(numbers) > 0 else 0
                wins = numbers[1] if len(numbers) > 1 else 0
                losses = numbers[2] if len(numbers) > 2 else 0
                win_rate = percentages[0] if percentages else (wins / total_games if total_games > 0 else 0)
                
                bans = 0
                if len(numbers) > 6:
                    bans = numbers[6] if len(numbers) > 6 else 0
                
                picks_bans = total_games + bans
                
                stats = {
                    'hero': hero_name,
                    'total_games': total_games,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'bans': bans,
                    'picks_bans': picks_bans,
                }
                
                hero_stats.append(stats)
            
            if hero_stats and is_main_stats:
                break
        
        if hero_stats:
            df = pd.DataFrame(hero_stats)
            df = df.sort_values('total_games', ascending=False).reset_index(drop=True)
            print(f"  Found {len(df)} heroes")
            return df
        
        print("  No hero statistics found")
        return None
    
    def scrape_tournament(self, url: str, include_details: bool = True) -> Dict:
        """Scrape all available data from a tournament."""
        print(f"\nScraping: {url}")
        print("-" * 50)
        
        result = {
            'url': url,
            'hero_statistics': None,
            'hero_synergies': [],
            'hero_counters': [],
        }
        
        hero_stats = self.scrape_hero_statistics(url)
        if hero_stats is not None:
            result['hero_statistics'] = hero_stats.to_dict('records')
            
            if include_details:
                synergies, counters = self.scrape_hero_details(url)
                result['hero_synergies'] = synergies
                result['hero_counters'] = counters
        
        return result
    
    def scrape_hero_details(self, url_or_title: str) -> Tuple[List[Dict], List[Dict]]:
        """Scrape detailed hero synergy and counter data from tournament page."""
        if url_or_title.startswith('http'):
            page_title = self.url_to_page_title(url_or_title)
        else:
            page_title = url_or_title
        
        if not page_title.endswith('Statistics') and not page_title.endswith('statistics'):
            page_title = page_title + '/Statistics'
        
        html = self.get_page_html(page_title)
        if not html:
            return [], []
        
        soup = BeautifulSoup(html, 'html.parser')
        
        synergies = []
        counters = []
        
        played_with_captions = soup.find_all('caption', string='Played With')
        played_against_captions = soup.find_all('caption', string='Played Against')
        
        for caption in played_with_captions:
            table = caption.find_parent('table')
            if not table:
                continue
            
            hero_name = self._find_hero_for_table(table)
            if hero_name:
                synergy_data = self._parse_detail_table(table, hero_name, 'synergy')
                synergies.extend(synergy_data)
        
        for caption in played_against_captions:
            table = caption.find_parent('table')
            if not table:
                continue
            
            hero_name = self._find_hero_for_table(table)
            if hero_name:
                counter_data = self._parse_detail_table(table, hero_name, 'counter')
                counters.extend(counter_data)
        
        print(f"  Found {len(synergies)} synergy records, {len(counters)} counter records")
        return synergies, counters
    
    def _find_hero_for_table(self, table) -> Optional[str]:
        """Find the hero name that a detail table belongs to."""
        parent_td = table.find_parent('td')
        if not parent_td:
            return None
        
        parent_tr = parent_td.find_parent('tr')
        if not parent_tr:
            return None
        
        hero_links = parent_tr.find_all('a', href=re.compile(r'^/mobilelegends/[^/]+$'))
        for link in hero_links:
            href = link.get('href', '')
            if '/File:' in href or '/Category:' in href:
                continue
            text = link.get('title', '') or link.get_text(strip=True)
            if text and len(text) > 1 and text.lower() not in self.NON_HEROES:
                return text
        
        return None
    
    def _parse_detail_table(self, table, hero_name: str, data_type: str) -> List[Dict]:
        """Parse a Played With or Played Against table."""
        results = []
        
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 4:
                continue
            
            partner_name = None
            partner_cell_idx = -1
            for idx, cell in enumerate(cells[:3]):
                link = cell.find('a')
                if link:
                    title = link.get('title', '') or link.get_text(strip=True)
                    if title and not title.startswith('File:'):
                        partner_name = title
                        partner_cell_idx = idx
                        break
            
            if not partner_name:
                continue
            
            numbers = []
            win_rate = 0.0
            
            for cell in cells[partner_cell_idx + 1:]:
                text = cell.get_text(strip=True)
                if '%' in text:
                    win_rate = self._parse_percent(text)
                elif text.isdigit():
                    numbers.append(int(text))
            
            if len(numbers) >= 3:
                games = numbers[0]
                wins = numbers[1]
                losses = numbers[2]
            elif len(numbers) >= 1:
                games = numbers[0]
                wins = 0
                losses = 0
            else:
                continue
            
            if win_rate == 0 and games > 0 and wins > 0:
                win_rate = wins / games
            
            record = {
                'hero': hero_name,
                'partner' if data_type == 'synergy' else 'opponent': partner_name,
                'games': games,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
            }
            
            results.append(record)
        
        return results
    
    def scrape_multiple(self, urls: List[str], output_dir: str = 'liquipedia_data', 
                        include_details: bool = False) -> dict:
        """Scrape multiple tournaments and combine the data."""
        os.makedirs(output_dir, exist_ok=True)
        
        all_stats = []
        all_synergies = []
        all_counters = []
        
        for url in urls:
            try:
                data = self.scrape_tournament(url, include_details=include_details)
                
                if data['hero_statistics']:
                    for stat in data['hero_statistics']:
                        stat['source'] = url
                        all_stats.append(stat)
                
                if include_details:
                    for syn in data.get('hero_synergies', []):
                        syn['source'] = url
                        all_synergies.append(syn)
                    for cnt in data.get('hero_counters', []):
                        cnt['source'] = url
                        all_counters.append(cnt)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        result = {'hero_statistics': pd.DataFrame(), 'hero_synergies': pd.DataFrame(), 
                  'hero_counters': pd.DataFrame()}
        
        if all_stats:
            df = pd.DataFrame(all_stats)
            result['hero_statistics'] = df
            
            csv_path = os.path.join(output_dir, 'hero_statistics.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nsaved {len(df)} hero statistics to {csv_path}")
            
            json_path = os.path.join(output_dir, 'hero_statistics.json')
            with open(json_path, 'w') as f:
                json.dump(all_stats, f, indent=2)
        
        if all_synergies:
            df_syn = pd.DataFrame(all_synergies)
            result['hero_synergies'] = df_syn
            
            csv_path = os.path.join(output_dir, 'hero_synergies.csv')
            df_syn.to_csv(csv_path, index=False)
            print(f"saved {len(df_syn)} synergy records to {csv_path}")
        
        if all_counters:
            df_cnt = pd.DataFrame(all_counters)
            result['hero_counters'] = df_cnt
            
            csv_path = os.path.join(output_dir, 'hero_counters.csv')
            df_cnt.to_csv(csv_path, index=False)
            print(f"saved {len(df_cnt)} counter records to {csv_path}")
        
        return result
    
    def _parse_percent(self, text: str) -> float:
        """Parse percentage from text."""
        try:
            clean = re.sub(r'[^\d.]', '', text)
            value = float(clean) if clean else 0
            if value > 1:
                value = value / 100
            return value
        except ValueError:
            return 0.0
    
    def aggregate_hero_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate hero statistics across multiple tournaments."""
        if df.empty:
            return df
        
        agg = df.groupby('hero').agg({
            'total_games': 'sum',
            'wins': 'sum',
            'losses': 'sum',
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['total_games'].replace(0, 1)
        agg['tournaments'] = df.groupby('hero')['source'].nunique().values
        
        return agg.sort_values('total_games', ascending=False)
    
    def aggregate_synergy_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate synergy statistics across multiple tournaments."""
        if df.empty:
            return df
        
        agg = df.groupby(['hero', 'partner']).agg({
            'games': 'sum',
            'wins': 'sum',
            'losses': 'sum',
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['games'].replace(0, 1)
        agg['tournaments'] = df.groupby(['hero', 'partner'])['source'].nunique().values
        
        return agg.sort_values(['hero', 'games'], ascending=[True, False])
    
    def aggregate_counter_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate counter statistics across multiple tournaments."""
        if df.empty:
            return df
        
        agg = df.groupby(['hero', 'opponent']).agg({
            'games': 'sum',
            'wins': 'sum',
            'losses': 'sum',
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['games'].replace(0, 1)
        agg['tournaments'] = df.groupby(['hero', 'opponent'])['source'].nunique().values
        
        return agg.sort_values(['hero', 'games'], ascending=[True, False])


def get_tournament_list() -> List[str]:
    """Get a list of recent major MLBB tournaments."""
    return [
        "https://liquipedia.net/mobilelegends/MPL/Indonesia/Season_14",
        "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14", 
        "https://liquipedia.net/mobilelegends/MPL/Malaysia/Season_14",
        "https://liquipedia.net/mobilelegends/M5_World_Championship",
        "https://liquipedia.net/mobilelegends/IESF/World_Esports_Championships/2025/Women",
        "https://liquipedia.net/mobilelegends/MPL/LATAM/Season_4",
        "https://liquipedia.net/mobilelegends/MPL/Brazil/Season_5",
    ]


def interactive_mode():
    """Interactive mode for scraping tournaments."""
    print("-" * 50)
    print("Liquipedia MLBB Scraper")
    print("-" * 50)
    
    print("\nOptions:")
    print("  1. Scrape a single tournament")
    print("  2. Scrape all major tournaments")
    print("  3. Enter custom URL")
    print("  4. Scrape with hero details (synergy/counter data)")
    
    choice = input("\nSelect option: ").strip()
    
    scraper = LiquipediaScraper(delay=2.0)
    
    if choice == '1':
        tournaments = get_tournament_list()
        print("\nAvailable tournaments:")
        for i, url in enumerate(tournaments, 1):
            name = url.split('/mobilelegends/')[-1]
            print(f"  {i}. {name}")
        
        idx = input("\nSelect tournament number: ").strip()
        try:
            url = tournaments[int(idx) - 1]
            data = scraper.scrape_tournament(url)
            
            if data['hero_statistics']:
                df = pd.DataFrame(data['hero_statistics'])
                print(f"\nTop 10 heroes by games played:")
                print(df.nlargest(10, 'total_games').to_string(index=False))
        except (ValueError, IndexError):
            print("Invalid selection")
    
    elif choice == '2':
        print("\nScraping all tournaments (this may take a while)...")
        result = scraper.scrape_multiple(get_tournament_list())
        
        if not result['hero_statistics'].empty:
            agg = scraper.aggregate_hero_stats(result['hero_statistics'])
            print(f"\nAggregated statistics (top 15 heroes):")
            print(agg.head(15).to_string(index=False))
    
    elif choice == '3':
        print("\nEnter Liquipedia URL (e.g., https://liquipedia.net/mobilelegends/MPL/Indonesia/Season_14):")
        url = input("> ").strip()
        
        if url:
            data = scraper.scrape_tournament(url)
            
            if data['hero_statistics']:
                df = pd.DataFrame(data['hero_statistics'])
                print(f"\nFound {len(df)} heroes:")
                print(df.to_string(index=False))
    
    elif choice == '4':
        print("\nScraping with hero details (synergy/counter data)...")
        print("This will take longer as it fetches detailed stats for each hero.\n")
        
        tournaments = get_tournament_list()
        print("Available tournaments:")
        for i, url in enumerate(tournaments, 1):
            name = url.split('/mobilelegends/')[-1]
            print(f"  {i}. {name}")
        print(f"  {len(tournaments) + 1}. All tournaments")
        
        idx = input("\nSelect option: ").strip()
        try:
            idx_num = int(idx)
            if idx_num == len(tournaments) + 1:
                urls = tournaments
            else:
                urls = [tournaments[idx_num - 1]]
            
            result = scraper.scrape_multiple(urls, include_details=True)
            
            if not result['hero_synergies'].empty:
                agg_syn = scraper.aggregate_synergy_stats(result['hero_synergies'])
                print(f"\ntop synergy pairs (by games played):")
                print(agg_syn.head(20).to_string(index=False))
            
            if not result['hero_counters'].empty:
                agg_cnt = scraper.aggregate_counter_stats(result['hero_counters'])
                print(f"\ntop counter matchups (by games played):")
                print(agg_cnt.head(20).to_string(index=False))
                
        except (ValueError, IndexError):
            print("Invalid selection")
    
    else:
        print("Invalid option")


if __name__ == "__main__":
    interactive_mode()
