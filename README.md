# mlbb-heropicker

Hero recommendation for mobile legends based on your position, teammates, and enemies.

Uses match data from kaggle + pro play stats from liquipedia to calculate synergy and counter scores.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Then just answer the prompts:
- position (jungle/roam/mid/exp/gold)
- teammates' heroes
- enemy heroes
- banned heroes

## data

the model uses:
- [kaggle mlbb match results](https://www.kaggle.com/datasets/rizqinur/mobile-legends-match-results) - 10k ranked matches
- liquipedia pro play statistics (...scraped with `liquipedia_scraper.py`)

to scrape more pro data:
```bash
python liquipedia_scraper.py
```

After adding new data, delete `mlbb_model.pkl` and run again to retrain.

## How it works

score = 35% synergy + 40% win rate + 10% pick rate + 15% counter

- **synergy**: how well heroes perform together (from match data + pro play)
- **win rate**: hero's win rate in your position
- **pick rate**: how often the hero is picked
- **counter**: how well the hero does against enemy picks (from pro play)
