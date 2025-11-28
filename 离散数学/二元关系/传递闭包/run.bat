@echo off
python ip2id.py input.csv
python adj2npy.py
python warshall.py
python reachable2csv.py
echo done 
pause