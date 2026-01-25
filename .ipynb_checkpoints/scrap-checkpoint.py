import argparse
from huggingface_hub import HfApi, snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Upload/Download from HuggingFace")
    parser.add_argument("--action", required=True, choices=["upload", "download"],
                        help="Action to perform: upload or download")
    parser.add_argument("--repo_id", default="Maheep/obfuscation", 
                        help="HuggingFace repo ID (e.g., Maheep/obfuscation)")
    parser.add_argument("--local_dir", default="safetynet/safetynet",
                        help="Local directory path")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"],
                        help="Repository type")
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()
    
    if args.action == "upload":
        print(f"Creating repo {args.repo_id}...")
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
        
        print(f"Uploading {args.local_dir} to {args.repo_id}...")
        api.upload_large_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            print_report=True,
        )
        print("Upload complete!")
        
    elif args.action == "download":
        print(f"Downloading {args.repo_id} to {args.local_dir}...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("Download complete!")


if __name__ == "__main__":
    main()
