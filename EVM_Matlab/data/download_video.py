import os
import requests

# 视频 URL 列表
video_urls = [
     "https://people.csail.mit.edu/mrub/evm/video/baby.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/baby2.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/face.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/face2.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/guitar.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/subway.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/shadow.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/camera.mp4",
     "https://people.csail.mit.edu/mrub/evm/video/wrist.mp4",
]

# 下载目录
download_path = "./data"

# 创建下载目录（如果不存在）
os.makedirs(download_path, exist_ok=True)

def download_video(url, path):
    local_filename = os.path.join(path, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# 下载所有视频
for url in video_urls:
    print(f"Downloading {url}...")
    download_video(url, download_path)
    print(f"Downloaded {url}")

print("All videos downloaded.")