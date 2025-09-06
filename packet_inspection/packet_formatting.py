import pandas as pd

df = pd.read_csv("datasets/csic_dataset.csv")
df_anomalous = df[df["Label"] == "Anomalous"]
header_map = {
    "User-Agent": "User-Agent",
    "Pragma": "Pragma",
    "Cache-Control": "Cache-Control",
    "Accept": "Accept",
    "Accept-encoding": "Accept-Encoding",
    "Accept-charset": "Accept-Charset",
    "language": "Accept-Language",
    "host": "Host",
    "cookie": "Cookie",
    "content-type": "Content-Type",
    "connection": "Connection",
    "lenght": "Content-Length"
}

import json
import random

def random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

with open("anomalous_packets.jsonl", "w") as f:
    for idx, row in df_anomalous.iterrows():
        ip = random_ip()
        request_line = f"{row['Method']} {row['URL']} HTTP/1.1"
        headers = []
        for col, header in header_map.items():
            if col in row and pd.notnull(row[col]):
                headers.append(f"{header}: {row[col]}")
        http_packet = ip + "\n" + request_line + "\n" + "\n".join(headers)
        obj = {"text": http_packet}
        f.write(json.dumps(obj) + "\n")