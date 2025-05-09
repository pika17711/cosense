tar -czvf cosense.tar.gz ../cosense
scp cosense.tar.gz nvidia@10.112.8.113:/home/nvidia/gzh
ssh nvidia@10.112.8.113 "tar -zxvf /home/nvidia/gzh/cosense.tar.gz -C /home/nvidia/gzh"