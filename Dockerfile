FROM registry.cn-shanghai.aliyuncs.com/tianchi_tky/pdf_query:1.2.5
COPY app /app
WORKDIR /app
RUN pip install pypdf
RUN pip install sentencepiece
RUN pip install jieba
CMD ["bash", "run.sh"]
