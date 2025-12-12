#!/usr/bin/env python3
"""
Script to download and setup the Kaggle MLBB dataset.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Setup API credentials: https://www.kaggle.com/docs/api
   - Go to kaggle.com -> Settings -> API -> Create New Token
   - Place kaggle.json in ~/.kaggle/
"""

import os
import sys
import subprocess
import zipfile


def check_kaggle_installed():
    """Check if kaggle CLI is installed."""
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_dataset():
    """Download the MLBB dataset from Kaggle."""
    dataset = 'rizqinur/mobile-legends-match-results'
    output_dir = 'data'
    
    print("🎮 MLBB Dataset Downloader")
    print("=" * 50)
    
    # Check kaggle installation
    if not check_kaggle_installed():
        print("❌ Kaggle CLI not found!")
        print("\n📦 Install with: pip install kaggle")
        print("\n📋 Then setup API credentials:")
        print("   1. Go to kaggle.com -> Settings -> API -> Create New Token")
        print("   2. Place the downloaded kaggle.json in ~/.kaggle/")
        print("   3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📥 Downloading {dataset}...")
    
    try:
        # Download dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset, '-p', output_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return False
        
        print("✅ Download complete!")
        
        # Extract zip file
        zip_path = os.path.join(output_dir, 'mobile-legends-match-results.zip')
        if os.path.exists(zip_path):
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            print("✅ Extraction complete!")
        
        # List downloaded files
        print("\n📁 Downloaded files:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = '  ' * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        
        print("\n✅ Dataset ready! Run main.py to start using it.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def manual_download_instructions():
    """Show manual download instructions."""
    print("\n" + "=" * 50)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 50)
    print("""
If automatic download doesn't work, follow these steps:

1. Go to: https://www.kaggle.com/datasets/rizqinur/mobile-legends-match-results

2. Click the "Download" button (you may need to sign in)

3. Extract the downloaded ZIP file

4. Create this folder structure in your project:
   
   mlbb-heropicker/
   └── data/
       ├── 1.7.58/
       │   ├── heroes.csv
       │   └── results.csv
       └── 1.7.68/
           ├── heroes.csv
           └── results.csv

5. Run: python main.py
""")


if __name__ == "__main__":
    success = download_dataset()
    
    if not success:
        manual_download_instructions()
        sys.exit(1)
