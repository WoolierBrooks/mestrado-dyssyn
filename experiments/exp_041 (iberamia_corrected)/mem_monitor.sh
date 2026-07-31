#!/bin/bash
# Roda em paralelo ao exp.py e loga memória/swap a cada 10s.
# Se o exp.py morrer por OOM, as últimas linhas antes do fim vão
# mostrar memória disponível caindo perto de zero.
#
# Uso:
#   nohup bash mem_monitor.sh > mem.log 2>&1 &

while true; do
    echo "----- $(date '+%Y-%m-%d %H:%M:%S') -----"
    free -h
    echo ""
    sleep 10
done