#!/bin/bash
docker login --username=toothfairy42 --password=wszwsz123123 registry.cn-shanghai.aliyuncs.com
docker build -t registry.cn-shanghai.aliyuncs.com/tianchi_tky/pdf_query:1.2.5 .