"""
MLBB Hero Picker - Machine Learning Based Hero Recommendation System
"""

import os
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

POSITIONS = ['jungle', 'roam', 'mid', 'exp', 'gold']

POSITION_ALIASES = {
    'jungler': 'jungle',
    'jg': 'jungle',
    'roamer': 'roam',
    'support': 'roam',
    'tank': 'roam',
    'midlane': 'mid',
    'midlaner': 'mid',
    'mage': 'mid',
    'explaner': 'exp',
    'exp lane': 'exp',
    'offlane': 'exp',
    'fighter': 'exp',
    'goldlane': 'gold',
    'gold lane': 'gold',
    'marksman': 'gold',
    'mm': 'gold',
    'adc': 'gold',
}

ROLE_TO_POSITION = {
    'Jungler': 'jungle',
    'Assassin': 'jungle',
    'Tank': 'roam',
    'Support': 'roam',
    'Mage': 'mid',
    'Fighter': 'exp',
    'Marksman': 'gold',
}


class LiquipediaDataLoader:
    """Load hero synergy and counter data scraped from Liquipedia."""
    
    def __init__(self, data_dir: str = 'liquipedia_data'):
        self.data_dir = data_dir
        self.synergies_df = None
        self.counters_df = None
        self.hero_stats_df = None
    
    def load_data(self) -> bool:
        """Load all available Liquipedia data."""
        synergy_path = os.path.join(self.data_dir, 'hero_synergies.csv')
        counter_path = os.path.join(self.data_dir, 'hero_counters.csv')
        stats_path = os.path.join(self.data_dir, 'hero_statistics.csv')
        
        loaded = False
        
        if os.path.exists(synergy_path):
            self.synergies_df = pd.read_csv(synergy_path)
            loaded = True
        
        if os.path.exists(counter_path):
            self.counters_df = pd.read_csv(counter_path)
            loaded = True
        
        if os.path.exists(stats_path):
            self.hero_stats_df = pd.read_csv(stats_path)
            loaded = True
        
        return loaded
    
    def get_synergy_matrix(self) -> Dict[Tuple[str, str], Dict]:
        """Build synergy lookup from Liquipedia data."""
        synergy_lookup = {}
        
        if self.synergies_df is None:
            return synergy_lookup
        
        agg = self.synergies_df.groupby(['hero', 'partner']).agg({
            'games': 'sum',
            'wins': 'sum',
            'losses': 'sum'
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['games'].replace(0, 1)
        
        for _, row in agg.iterrows():
            hero = row['hero']
            partner = row['partner']
            key = tuple(sorted([hero, partner]))
            
            if key not in synergy_lookup or row['games'] > synergy_lookup[key]['games']:
                synergy_lookup[key] = {
                    'games': row['games'],
                    'wins': row['wins'],
                    'win_rate': row['win_rate']
                }
        
        return synergy_lookup
    
    def get_counter_matrix(self) -> Dict[Tuple[str, str], Dict]:
        """Build counter lookup from Liquipedia data."""
        counter_lookup = {}
        
        if self.counters_df is None:
            return counter_lookup
        
        agg = self.counters_df.groupby(['hero', 'opponent']).agg({
            'games': 'sum',
            'wins': 'sum',
            'losses': 'sum'
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['games'].replace(0, 1)
        
        for _, row in agg.iterrows():
            hero = row['hero']
            opponent = row['opponent']
            
            counter_lookup[(hero, opponent)] = {
                'games': row['games'],
                'wins': row['wins'],
                'win_rate': row['win_rate']
            }
        
        return counter_lookup
    
    def get_hero_pro_stats(self) -> Dict[str, Dict]:
        """Get aggregated pro play statistics per hero."""
        stats = {}
        
        if self.hero_stats_df is None:
            return stats
        
        agg = self.hero_stats_df.groupby('hero').agg({
            'total_games': 'sum',
            'wins': 'sum',
            'losses': 'sum'
        }).reset_index()
        
        agg['win_rate'] = agg['wins'] / agg['total_games'].replace(0, 1)
        
        for _, row in agg.iterrows():
            stats[row['hero']] = {
                'games': row['total_games'],
                'wins': row['wins'],
                'win_rate': row['win_rate']
            }
        
        return stats


class DataLoader:
    """Load and preprocess MLBB match data from Kaggle dataset."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.heroes_df = None
        self.results_df = None
        
    def load_data(self, patch: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load heroes and match results data."""
        patches = ['1.7.58', '1.7.68'] if patch == 'all' else [patch]
        
        heroes_list = []
        results_list = []
        
        for p in patches:
            patch_dir = os.path.join(self.data_dir, p)
            
            heroes_path = os.path.join(patch_dir, 'heroes.csv')
            results_path = os.path.join(patch_dir, 'results.csv')
            
            if os.path.exists(heroes_path):
                heroes_df = pd.read_csv(heroes_path)
                heroes_df['patch'] = p
                heroes_list.append(heroes_df)
                
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                results_df['patch'] = p
                results_list.append(results_df)
        
        if heroes_list:
            self.heroes_df = pd.concat(heroes_list, ignore_index=True)
            if 'name' in self.heroes_df.columns:
                self.heroes_df = self.heroes_df.drop_duplicates(subset=['name'], keep='last')
            elif 'hero' in self.heroes_df.columns:
                self.heroes_df = self.heroes_df.drop_duplicates(subset=['hero'], keep='last')
        
        if results_list:
            self.results_df = pd.concat(results_list, ignore_index=True)
            self.results_df = self.results_df.drop_duplicates(subset=['battle_id'], keep='first')
        
        return self.heroes_df, self.results_df
    
    def get_hero_positions(self) -> Dict[str, List[str]]:
        """Extract hero-to-position mapping from heroes data."""
        hero_positions = defaultdict(list)
        
        if self.heroes_df is not None and 'lane' in self.heroes_df.columns:
            for _, row in self.heroes_df.iterrows():
                hero = row.get('name', row.get('hero', None))
                if hero is None or pd.isna(hero):
                    continue
                hero = str(hero).strip()
                
                lane = row.get('lane', '')
                role = row.get('roles', row.get('role', ''))
                
                if pd.notna(lane):
                    lane_str = str(lane).lower()
                    if 'gold' in lane_str:
                        hero_positions[hero].append('gold')
                    elif 'exp' in lane_str:
                        hero_positions[hero].append('exp')
                    elif 'jungle' in lane_str:
                        hero_positions[hero].append('jungle')
                    elif 'mid' in lane_str:
                        hero_positions[hero].append('mid')
                    elif 'roam' in lane_str:
                        hero_positions[hero].append('roam')
                    else:
                        lanes = lane_str.split(',')
                        for l in lanes:
                            l = l.strip()
                            if l in POSITIONS:
                                hero_positions[hero].append(l)
                            elif l in POSITION_ALIASES:
                                hero_positions[hero].append(POSITION_ALIASES[l])
                
                if not hero_positions[hero] and pd.notna(role):
                    roles = str(role).replace('/', ',').split(',')
                    for r in roles:
                        r = r.strip()
                        if r in ROLE_TO_POSITION:
                            pos = ROLE_TO_POSITION[r]
                            if pos not in hero_positions[hero]:
                                hero_positions[hero].append(pos)
        
        return dict(hero_positions)
    
    def extract_team_compositions(self) -> List[Dict]:
        """Extract team compositions from match results."""
        compositions = []
        
        if self.results_df is None:
            return compositions
        
        for _, row in self.results_df.iterrows():
            match_id = row.get('battle_id', row.name)
            
            left_heroes = []
            right_heroes = []
            
            if 'left_heroes' in row.index and pd.notna(row['left_heroes']):
                try:
                    left_heroes = ast.literal_eval(str(row['left_heroes']))
                except (ValueError, SyntaxError):
                    left_heroes = []
            
            if 'right_heroes' in row.index and pd.notna(row['right_heroes']):
                try:
                    right_heroes = ast.literal_eval(str(row['right_heroes']))
                except (ValueError, SyntaxError):
                    right_heroes = []
            
            if not left_heroes:
                for i in range(1, 6):
                    col1 = f'team1_hero{i}'
                    col2 = f'team2_hero{i}'
                    if col1 in row.index and pd.notna(row[col1]):
                        left_heroes.append(str(row[col1]).strip())
                    if col2 in row.index and pd.notna(row[col2]):
                        right_heroes.append(str(row[col2]).strip())
            
            match_result = row.get('match_result', row.get('winner', row.get('result', None)))
            
            left_won = False
            right_won = False
            
            if pd.notna(match_result):
                result_str = str(match_result).lower()
                if result_str in ['victory', 'win', '1', 'team1']:
                    left_won = True
                elif result_str in ['defeat', 'lose', '2', 'team2']:
                    right_won = True
            
            if len(left_heroes) == 5:
                compositions.append({
                    'team': [str(h).strip() for h in left_heroes],
                    'won': left_won,
                    'match_id': match_id
                })
            
            if len(right_heroes) == 5:
                compositions.append({
                    'team': [str(h).strip() for h in right_heroes],
                    'won': right_won,
                    'match_id': match_id
                })
        
        return compositions


class HeroSynergyModel:
    """Learn hero synergies from match data using co-occurrence win rates."""
    
    def __init__(self):
        self.hero_pair_wins = defaultdict(int)
        self.hero_pair_games = defaultdict(int)
        self.hero_wins = defaultdict(int)
        self.hero_games = defaultdict(int)
        self.hero_position_wins = {}
        self.hero_position_games = {}
        self.all_heroes = set()
        self.synergy_matrix = None
        self.liquipedia_synergy = {}
        self.liquipedia_counters = {}
        self.liquipedia_stats = {}
    
    def _get_position_dict(self, hero: str, wins: bool = False) -> dict:
        """Get or create position dict for a hero."""
        storage = self.hero_position_wins if wins else self.hero_position_games
        if hero not in storage:
            storage[hero] = defaultdict(int)
        return storage[hero]
        
    def fit(self, compositions: List[Dict], hero_positions: Dict[str, List[str]] = None,
             liquipedia_synergy: Dict = None, liquipedia_counters: Dict = None,
             liquipedia_stats: Dict = None):
        """Train the model on team compositions."""
        self.liquipedia_synergy = liquipedia_synergy or {}
        self.liquipedia_counters = liquipedia_counters or {}
        self.liquipedia_stats = liquipedia_stats or {}
        
        for hero in self.liquipedia_stats:
            self.all_heroes.add(hero)
        
        for comp in compositions:
            team = comp['team']
            won = comp['won']
            
            for hero in team:
                self.all_heroes.add(hero)
                self.hero_games[hero] += 1
                if won:
                    self.hero_wins[hero] += 1
                
                if hero_positions and hero in hero_positions:
                    for pos in hero_positions[hero]:
                        self._get_position_dict(hero, wins=False)[pos] += 1
                        if won:
                            self._get_position_dict(hero, wins=True)[pos] += 1
            
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    pair = tuple(sorted([team[i], team[j]]))
                    self.hero_pair_games[pair] += 1
                    if won:
                        self.hero_pair_wins[pair] += 1
        
        self._build_synergy_matrix()
        return self
    
    def _build_synergy_matrix(self):
        """Build the hero synergy matrix."""
        heroes = sorted(self.all_heroes)
        n = len(heroes)
        hero_to_idx = {h: i for i, h in enumerate(heroes)}
        
        total_wins = sum(self.hero_wins.values())
        total_games = sum(self.hero_games.values())
        base_win_rate = total_wins / total_games if total_games > 0 else 0.5
        
        self.synergy_matrix = np.zeros((n, n))
        self.hero_to_idx = hero_to_idx
        self.idx_to_hero = {i: h for h, i in hero_to_idx.items()}
        
        for pair, games in self.hero_pair_games.items():
            if games < 3:
                continue
                
            wins = self.hero_pair_wins[pair]
            win_rate = wins / games
            synergy = win_rate - base_win_rate
            
            i, j = hero_to_idx[pair[0]], hero_to_idx[pair[1]]
            self.synergy_matrix[i, j] = synergy
            self.synergy_matrix[j, i] = synergy
    
    def get_synergy(self, hero1: str, hero2: str) -> float:
        """Get synergy score between two heroes, combining Kaggle and Liquipedia data."""
        kaggle_synergy = 0.0
        if hero1 in self.hero_to_idx and hero2 in self.hero_to_idx:
            i, j = self.hero_to_idx[hero1], self.hero_to_idx[hero2]
            kaggle_synergy = self.synergy_matrix[i, j]
        
        lp_synergy = 0.0
        pair_key = tuple(sorted([hero1, hero2]))
        if pair_key in self.liquipedia_synergy:
            lp_data = self.liquipedia_synergy[pair_key]
            if lp_data['games'] >= 3:
                lp_synergy = lp_data['win_rate'] - 0.5
        
        if self.liquipedia_synergy and not self.hero_pair_games:
            return lp_synergy
        elif not self.liquipedia_synergy:
            return kaggle_synergy
        else:
            return 0.6 * kaggle_synergy + 0.4 * lp_synergy
    
    def get_counter_score(self, hero: str, opponent: str) -> float:
        """Get counter score when hero plays against opponent."""
        if (hero, opponent) in self.liquipedia_counters:
            data = self.liquipedia_counters[(hero, opponent)]
            if data['games'] >= 3:
                return data['win_rate'] - 0.5
        return 0.0
    
    def get_hero_win_rate(self, hero: str, position: str = None) -> float:
        """Get hero's overall or position-specific win rate."""
        if position and hero in self.hero_position_games:
            games = self.hero_position_games[hero].get(position, 0)
            wins = self.hero_position_wins[hero].get(position, 0)
            if games > 0:
                return wins / games
        
        games = self.hero_games.get(hero, 0)
        wins = self.hero_wins.get(hero, 0)
        return wins / games if games > 0 else 0.5
    
    def get_team_synergy(self, heroes: List[str]) -> float:
        """Calculate total synergy score for a team."""
        total_synergy = 0.0
        for i in range(len(heroes)):
            for j in range(i + 1, len(heroes)):
                total_synergy += self.get_synergy(heroes[i], heroes[j])
        return total_synergy
    
    def save(self, filepath: str):
        """Save model to file."""
        self._prepare_for_pickle()
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def _prepare_for_pickle(self):
        """Convert defaultdicts to regular dicts for pickling."""
        self.hero_pair_wins = dict(self.hero_pair_wins)
        self.hero_pair_games = dict(self.hero_pair_games)
        self.hero_wins = dict(self.hero_wins)
        self.hero_games = dict(self.hero_games)
        self.hero_position_wins = {k: dict(v) for k, v in self.hero_position_wins.items()}
        self.hero_position_games = {k: dict(v) for k, v in self.hero_position_games.items()}
    
    @classmethod
    def load(cls, filepath: str) -> 'HeroSynergyModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class HeroRecommender:
    """Recommend heroes based on position and team composition."""
    
    def __init__(self, synergy_model: HeroSynergyModel, 
                 hero_positions: Dict[str, List[str]] = None,
                 synergy_weight: float = 0.35,
                 winrate_weight: float = 0.4,
                 popularity_weight: float = 0.1,
                 counter_weight: float = 0.15):
        """Initialize the recommender with a trained synergy model."""
        self.synergy_model = synergy_model
        self.hero_positions = hero_positions or {}
        self.synergy_weight = synergy_weight
        self.winrate_weight = winrate_weight
        self.popularity_weight = popularity_weight
        self.counter_weight = counter_weight
        
        total_games = sum(synergy_model.hero_games.values())
        if total_games > 0:
            self.pick_rates = {
                hero: games / total_games 
                for hero, games in synergy_model.hero_games.items()
            }
        else:
            self.pick_rates = {hero: 0.01 for hero in synergy_model.all_heroes}
    
    def get_candidates(self, position: str, exclude: List[str] = None) -> List[str]:
        """Get candidate heroes for a position."""
        position = self._normalize_position(position)
        exclude = set(exclude or [])
        exclude = {h.lower() for h in exclude}
        
        candidates = []
        
        for hero in self.synergy_model.all_heroes:
            if hero.lower() in exclude:
                continue
            
            valid_positions = self.hero_positions.get(hero, POSITIONS)
            if position in valid_positions or not valid_positions:
                candidates.append(hero)
        
        return candidates
    
    def _normalize_position(self, position: str) -> str:
        """Normalize position name."""
        position = position.lower().strip()
        return POSITION_ALIASES.get(position, position)
    
    def _normalize_hero(self, hero: str) -> Optional[str]:
        """Find matching hero name (case-insensitive)."""
        hero_lower = hero.lower().strip()
        for h in self.synergy_model.all_heroes:
            if h.lower() == hero_lower:
                return h
        return None
    
    def recommend(self, position: str, teammates: List[str], 
                  top_k: int = 5, exclude: List[str] = None,
                  enemies: List[str] = None) -> List[Dict]:
        """Recommend heroes for a position given current teammates and enemies."""
        position = self._normalize_position(position)
        
        normalized_teammates = []
        for t in teammates:
            norm = self._normalize_hero(t)
            if norm:
                normalized_teammates.append(norm)
        
        normalized_enemies = []
        if enemies:
            for e in enemies:
                norm = self._normalize_hero(e)
                if norm:
                    normalized_enemies.append(norm)
        
        all_exclude = list(set(teammates + (exclude or [])))
        candidates = self.get_candidates(position, all_exclude)
        
        if not candidates:
            return []
        
        scores = []
        max_pick_rate = max(self.pick_rates.values()) if self.pick_rates else 1
        
        for hero in candidates:
            synergy_score = 0
            for teammate in normalized_teammates:
                synergy_score += self.synergy_model.get_synergy(hero, teammate)
            
            if normalized_teammates:
                synergy_score /= len(normalized_teammates)
            
            counter_score = 0
            if normalized_enemies:
                for enemy in normalized_enemies:
                    counter_score += self.synergy_model.get_counter_score(hero, enemy)
                counter_score /= len(normalized_enemies)
            
            win_rate = self.synergy_model.get_hero_win_rate(hero, position)
            pick_rate = self.pick_rates.get(hero, 0) / max_pick_rate
            
            total_score = (
                self.synergy_weight * synergy_score +
                self.winrate_weight * (win_rate - 0.5) +
                self.popularity_weight * pick_rate +
                self.counter_weight * counter_score
            )
            
            scores.append({
                'hero': hero,
                'score': total_score,
                'synergy': synergy_score,
                'counter': counter_score,
                'win_rate': win_rate,
                'pick_rate': self.pick_rates.get(hero, 0),
                'position': position
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def explain_recommendation(self, hero: str, teammates: List[str]) -> str:
        """Generate human-readable explanation for a recommendation."""
        hero = self._normalize_hero(hero)
        if not hero:
            return "couldn't find that hero"
        
        lines = [f"{hero}:"]
        
        win_rate = self.synergy_model.get_hero_win_rate(hero)
        lines.append(f"  win rate: {win_rate:.0%}")
        
        if teammates:
            lines.append(f"  synergy:")
            for teammate in teammates:
                norm_teammate = self._normalize_hero(teammate)
                if norm_teammate:
                    synergy = self.synergy_model.get_synergy(hero, norm_teammate)
                    indicator = "+" if synergy > 0 else "-" if synergy < 0 else "~"
                    lines.append(f"    {indicator} {norm_teammate}: {synergy:+.3f}")
        
        pick_rate = self.pick_rates.get(hero, 0)
        lines.append(f"  pick rate: {pick_rate:.1%}")
        
        return "\n".join(lines)


def generate_sample_data() -> Tuple[Dict[str, List[str]], List[Dict]]:
    """Generate sample MLBB data for testing when no dataset is available."""
    hero_positions = {
        'Fanny': ['jungle'], 'Ling': ['jungle'], 'Lancelot': ['jungle'],
        'Hayabusa': ['jungle'], 'Gusion': ['jungle'], 'Karina': ['jungle'],
        'Aamon': ['jungle'], 'Akai': ['jungle', 'roam'],
        'Khufra': ['roam'], 'Atlas': ['roam'], 'Tigreal': ['roam'],
        'Franco': ['roam'], 'Chou': ['roam', 'exp'], 'Rafaela': ['roam'],
        'Estes': ['roam'], 'Mathilda': ['roam', 'jungle'],
        'Yve': ['mid'], 'Pharsa': ['mid'], 'Kagura': ['mid'],
        'Lunox': ['mid'], 'Cecilion': ['mid'], 'Valentina': ['mid'],
        'Xavier': ['mid'], 'Lylia': ['mid'],
        'Esmeralda': ['exp'], 'Yu Zhong': ['exp'], 'Paquito': ['exp'],
        'Terizla': ['exp'], 'Dyrroth': ['exp'], 'Thamuz': ['exp'],
        'Uranus': ['exp'], 'Grock': ['exp', 'roam'],
        'Beatrix': ['gold'], 'Brody': ['gold'], 'Claude': ['gold'],
        'Wanwan': ['gold'], 'Karrie': ['gold'], 'Moskov': ['gold'],
        'Miya': ['gold'], 'Bruno': ['gold'],
    }
    
    compositions = []
    np.random.seed(42)
    
    for i in range(1000):
        team = []
        for pos in POSITIONS:
            pos_heroes = [h for h, positions in hero_positions.items() 
                         if pos in positions and h not in team]
            if pos_heroes:
                team.append(np.random.choice(pos_heroes))
        
        base_win_prob = 0.5
        
        if 'Khufra' in team and 'Gusion' in team:
            base_win_prob += 0.05
        if 'Atlas' in team and 'Pharsa' in team:
            base_win_prob += 0.06
        if 'Estes' in team:
            base_win_prob += 0.03
        if 'Beatrix' in team:
            base_win_prob += 0.02
        
        won = np.random.random() < base_win_prob
        
        compositions.append({
            'team': team,
            'won': won,
            'match_id': f'sample_{i}'
        })
    
    return hero_positions, compositions


class MLBBHeroPicker:
    """Main interface for the MLBB Hero Picker system."""
    
    def __init__(self):
        self.data_loader = None
        self.liquipedia_loader = None
        self.synergy_model = None
        self.recommender = None
        self.hero_positions = None
        self.is_trained = False
        self.liquipedia_synergy = {}
        self.liquipedia_counters = {}
        self.liquipedia_stats = {}
    
    def load_liquipedia_data(self, data_dir: str = 'liquipedia_data'):
        """Load Liquipedia pro play data."""
        self.liquipedia_loader = LiquipediaDataLoader(data_dir)
        if self.liquipedia_loader.load_data():
            self.liquipedia_synergy = self.liquipedia_loader.get_synergy_matrix()
            self.liquipedia_counters = self.liquipedia_loader.get_counter_matrix()
            self.liquipedia_stats = self.liquipedia_loader.get_hero_pro_stats()
            
            syn_count = len(self.liquipedia_synergy)
            cnt_count = len(self.liquipedia_counters)
            hero_count = len(self.liquipedia_stats)
            
            print(f"  {syn_count} synergy pairs, {cnt_count} counter matchups, {hero_count} heroes")
        else:
            print("  no liquipedia data found")
        
        return self
    
    def load_kaggle_data(self, data_dir: str = 'data'):
        """Load data from Kaggle dataset."""
        self.data_loader = DataLoader(data_dir)
        heroes_df, results_df = self.data_loader.load_data()
        
        if heroes_df is not None and results_df is not None:
            print(f"  {len(heroes_df)} heroes, {len(results_df)} matches")
        
        return self
    
    def use_sample_data(self):
        """Use generated sample data for testing."""
        self.hero_positions, compositions = generate_sample_data()
        
        print(f"  {len(compositions)} matches, {len(self.hero_positions)} heroes")
        
        self.synergy_model = HeroSynergyModel()
        self.synergy_model.fit(
            compositions, 
            self.hero_positions,
            self.liquipedia_synergy,
            self.liquipedia_counters,
            self.liquipedia_stats
        )
        
        self.recommender = HeroRecommender(
            self.synergy_model,
            self.hero_positions
        )
        
        self.is_trained = True
        return self
    
    def train(self):
        """Train the model on loaded data."""
        if self.data_loader is None:
            raise ValueError("no data loaded")
        
        self.hero_positions = self.data_loader.get_hero_positions()
        compositions = self.data_loader.extract_team_compositions()
        
        if not compositions:
            print("  no compositions found, check data format")
            return self
        
        print(f"  {len(compositions)} team comps, {len(self.hero_positions)} hero positions")
        
        self.synergy_model = HeroSynergyModel()
        self.synergy_model.fit(
            compositions, 
            self.hero_positions,
            self.liquipedia_synergy,
            self.liquipedia_counters,
            self.liquipedia_stats
        )
        
        self.recommender = HeroRecommender(
            self.synergy_model,
            self.hero_positions
        )
        
        self.is_trained = True
        print("  done")
        
        return self
    
    def recommend(self, position: str, teammates: List[str], 
                  top_k: int = 5, banned: List[str] = None,
                  enemies: List[str] = None) -> List[Dict]:
        """Get hero recommendations for a position given current teammates and enemies."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or use_sample_data() first.")
        
        return self.recommender.recommend(position, teammates, top_k, banned, enemies)
    
    def explain(self, hero: str, teammates: List[str]) -> str:
        """Get explanation for why a hero is recommended."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        return self.recommender.explain_recommendation(hero, teammates)
    
    def save_model(self, filepath: str = 'mlbb_model.pkl'):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No model to save.")
        
        self.synergy_model._prepare_for_pickle()
        
        data = {
            'synergy_model': self.synergy_model,
            'hero_positions': self.hero_positions,
            'liquipedia_synergy': self.liquipedia_synergy,
            'liquipedia_counters': self.liquipedia_counters,
            'liquipedia_stats': self.liquipedia_stats,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"  saved to {filepath}")
    
    def load_model(self, filepath: str = 'mlbb_model.pkl'):
        """Load previously trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.synergy_model = data['synergy_model']
        self.hero_positions = data['hero_positions']
        self.liquipedia_synergy = data.get('liquipedia_synergy', {})
        self.liquipedia_counters = data.get('liquipedia_counters', {})
        self.liquipedia_stats = data.get('liquipedia_stats', {})
        self.recommender = HeroRecommender(
            self.synergy_model,
            self.hero_positions
        )
        self.is_trained = True
        print(f"  loaded {filepath}")
        return self
    
    def interactive_mode(self):
        """Run interactive recommendation session."""
        print("\n" + "="*20)
        print("  MLBB Hero Picker")
        print("="*20)
        print("type 'q' to quit\n")
        
        while True:
            print("position? (jungle/roam/mid/exp/gold)")
            position = input("> ").strip()
            
            if position.lower() in ['quit', 'exit', 'q']:
                break
            
            if position.lower() not in POSITIONS and position.lower() not in POSITION_ALIASES:
                print(f"not a valid position. try: {', '.join(POSITIONS)}\n")
                continue
            
            print("teammates? (comma-separated, or just hit enter)")
            teammates_input = input("> ").strip()
            teammates = [t.strip() for t in teammates_input.split(',') if t.strip()]
            
            print("enemies? (comma-separated, or just hit enter)")
            enemies_input = input("> ").strip()
            enemies = [e.strip() for e in enemies_input.split(',') if e.strip()]
            
            print("bans? (comma-separated, or just hit enter)")
            banned_input = input("> ").strip()
            banned = [b.strip() for b in banned_input.split(',') if b.strip()]
            
            try:
                recommendations = self.recommend(
                    position, teammates, top_k=5, banned=banned, enemies=enemies
                )
                
                print(f"\n-- {position} picks --")
                
                for i, rec in enumerate(recommendations, 1):
                    counter_str = ""
                    if rec.get('counter', 0) != 0:
                        counter_str = f", vs enemy {rec['counter']:+.2f}"
                    print(f"{i}. {rec['hero']} [{rec['score']:.2f}] "
                          f"({rec['win_rate']:.0%} wr, synergy {rec['synergy']:+.2f}{counter_str})")
                
                if recommendations:
                    print("\nwant details on a hero? (or just hit enter)")
                    hero_input = input("> ").strip()
                    if hero_input:
                        print("\n" + self.explain(hero_input, teammates))
                        
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    picker = MLBBHeroPicker()
    
    if os.path.exists('mlbb_model.pkl'):
        print("loading model...")
        picker.load_model()
    else:
        if os.path.exists('liquipedia_data'):
            print("loading pro play data...")
            picker.load_liquipedia_data()
        
        if os.path.exists('data/1.7.58') or os.path.exists('data/1.7.68'):
            print("loading match data...")
            picker.load_kaggle_data()
            print("training...")
            picker.train()
            picker.save_model()
        else:
            print("no kaggle data found, using sample data")
            print("(download from kaggle.com/datasets/rizqinur/mobile-legends-match-results)")
            picker.use_sample_data()
    
    picker.interactive_mode()
