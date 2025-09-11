cd /root/wangmy/engine
go test ./datasource/base -run TestUpdateAllKLine -count=1
cd /root/wangmy/Kronos/examples
sed -i 's/datetime/timestamps/g' /root/.quant1x/5min/sz000/sz000776.csv
/usr/bin/python3  000776.py
rm -rf  /root/.quant1x/5min/sz000/sz000776.csv
/usr/bin/python3 -m http.server
