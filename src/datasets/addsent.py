#!/usr/bin/env python3
"""Download and process original AddSent/AddOneSent datasets"""

import json
import argparse
import requests
import os

# Original URLs from the paper
_URLS = {
    "AddSent": "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/",
    "AddOneSent": "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/",
}

def download_dataset(url: str, output_path: str):
    """Download dataset from URL"""
    print(f"Downloading from: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading: {e}")
        return False

def process_downloaded_data(input_path: str, output_path: str):
    """Process downloaded data into standard format"""
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing {len(data.get('data', []))} articles...")
        
        # Count total examples
        total_examples = 0
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                total_examples += len(paragraph.get('qas', []))
        
        print(f"Total examples: {total_examples}")
        
        # Save processed data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed data saved to: {output_path}")
        
        # Show sample
        if data.get('data') and data['data'][0].get('paragraphs'):
            sample_para = data['data'][0]['paragraphs'][0]
            if sample_para.get('qas'):
                sample_qa = sample_para['qas'][0]
                print(f"\nSample example:")
                print(f"Question: {sample_qa['question']}")
                print(f"Context: {sample_para['context'][:200]}...")
                if sample_qa.get('answers'):
                    print(f"Answer: {sample_qa['answers'][0]['text']}")
        
        return True
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing data: {e}")
        return False

def download_addsent_dataset(method: str = "AddSent", output_dir: str = "utils/data/dataset"):
    """Download original AddSent/AddOneSent dataset"""
    
    if method not in _URLS:
        print(f"Error: Method '{method}' not supported. Use 'AddSent' or 'AddOneSent'")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download raw data
    raw_file = os.path.join(output_dir, f"{method}_raw.json")
    final_file = os.path.join(output_dir, f"{method.lower()}.json")
    
    print(f"Downloading original {method} dataset...")
    
    if download_dataset(_URLS[method], raw_file):
        print(f"Processing {method} dataset...")
        if process_downloaded_data(raw_file, final_file):
            # Clean up raw file
            os.remove(raw_file)
            print(f"{method} dataset ready at: {final_file}")
            return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Download original AddSent/AddOneSent datasets")
    parser.add_argument("--method", choices=["AddSent", "AddOneSent"], default="AddSent",
                       help="Which dataset to download")
    parser.add_argument("--output_dir", default="utils/data/dataset",
                       help="Output directory")
    parser.add_argument("--both", action="store_true",
                       help="Download both AddSent and AddOneSent")
    
    args = parser.parse_args()
    
    if args.both:
        print("Downloading both datasets...")
        success1 = download_addsent_dataset("AddSent", args.output_dir)
        success2 = download_addsent_dataset("AddOneSent", args.output_dir)
        
        if success1 and success2:
            print("Both datasets downloaded successfully!")
        else:
            print("Some downloads failed.")
    else:
        success = download_addsent_dataset(args.method, args.output_dir)
        if success:
            print(f"{args.method} dataset downloaded successfully!")
        else:
            print("Download failed.")

if __name__ == "__main__":
    main()