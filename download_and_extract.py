import os
import urllib.request
import tarfile

# Define download URL and destination filename
url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
download_path = "20news-bydate.tar.gz"
extract_path = "20news-bydate"

# Download the dataset if not already downloaded
if not os.path.exists(download_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, download_path)
    print("Download complete.")
else:
    print("Dataset already downloaded.")

# Extract the .tar.gz file
print("Extracting contents...")
with tarfile.open(download_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
print(f"Extraction complete. Files extracted to '{extract_path}/'")
