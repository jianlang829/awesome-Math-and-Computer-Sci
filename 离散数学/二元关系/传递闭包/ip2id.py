import sys, csv, json

ip2id = {}


def get_id(ip):
    if ip not in ip2id:
        ip2id[ip] = len(ip2id)
    return ip2id[ip]


with open(sys.argv[1]) as f:
    for a, b in csv.reader(f):
        get_id(a)
        get_id(b)

json.dump(ip2id, open("ip2id.json", "w"), indent=2)
print("n =", len(ip2id), "个顶点")
