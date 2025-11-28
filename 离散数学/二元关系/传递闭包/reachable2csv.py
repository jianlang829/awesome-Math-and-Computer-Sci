import numpy as np, json, csv

ip2id = json.load(open("ip2id.json"))
id2ip = {v: k for k, v in ip2id.items()}
R = np.load("reachable.npy")
ips = [id2ip[i] for i in range(len(id2ip))]
writer = csv.writer(open("reachable.csv", "w"))
writer.writerow(["from_ip", "to_ip", "reachable"])
for i in range(len(ips)):
    for j in range(len(ips)):
        writer.writerow([ips[i], ips[j], int(R[i, j])])
