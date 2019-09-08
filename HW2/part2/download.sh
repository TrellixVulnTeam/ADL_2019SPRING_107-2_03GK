wget https://www.dropbox.com/s/ahfg9rp411rhhqr/BERT_best.bin
wget https://www.dropbox.com/s/qa2jcdstdcaqwu4/BERT_strong.bin
mv BERT_best.bin best
mv BERT_strong.bin strong

wget https://www.dropbox.com/s/myiwy29bemi9qqf/9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba

cp ./9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba ./best/.pytorch_pretrained_bert/distributed_-1/

mv ./9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba ./strong/.pytorch_pretrained_bert/distributed_-1/



