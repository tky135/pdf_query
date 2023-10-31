from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, SummaryIndex
from llama_index.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.llms.base import llm_completion_callback
from llama_index.prompts import PromptTemplate
import json
from typing import Any
import torch
from util import read_json, read_pdf
import jieba
import jieba.analyse
import jieba.posseg
import os
import sys
import tky_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_WINDOW = 2048
NUM_OUTPUT = 128
SPLIT_LENGTH = 200  # assuming the length of target text is 200, it must be contained in a single chunk

tokenizer = AutoTokenizer.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
model = AutoModelForCausalLM.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
log_file = open("log.tky", "w", encoding="utf-8")
model.half()

pipe = pipeline("text-generation", device=device, tokenizer=tokenizer, model=model)    # Use pipeline as a high level helper
# 1. BUILD INVERTED INDEX
# txt = read_pdf("data/dataset.pdf") if not os.path.exists("data/dataset.txt") else open("data/dataset.txt", "r", encoding="utf-8").read()
# # split txt into overlapping chunks of SPLIT_LENGTH characters
# # build inverted index for each chunk
# txt_list = []

# NOTE using preprocessed txt list
txt_list = tky_preprocess.get_chunks()
print(len(txt_list))
for i in txt_list:
    print(len(i), end=", ")
raise Exception("break")
inverted_index = {}
counter = 0
for i in range(0, len(txt), SPLIT_LENGTH // 2):  # TODO what sbout SPLIT_LENGTH // 2
    small_txt = txt[i:i+SPLIT_LENGTH] if i + SPLIT_LENGTH < len(txt) else txt[i:]
    # chinese_parition = jieba.cut(small_txt, cut_all=True)  # full mode
    chinese_parition = jieba.cut_for_search(small_txt)  # search engine mode
    for word in chinese_parition:
        if word not in inverted_index:
            # TODO do some filtering here
            if word.isdigit():
                continue
            inverted_index[word] = set()
        inverted_index[word].add(counter)
    txt_list.append(small_txt)
    counter += 1
if os.path.exists("chunks"):
    os.system("rm -rf chunks")
os.mkdir("chunks")
# write each txt chunk to file
for i in range(len(txt_list)):
    with open("chunks/%d.txt" % i, "w", encoding="utf-8") as f:
        print(txt_list[i], file=f)
# write inverted index to file
with open("inverted_index.txt", "w", encoding="utf-8") as f:
    for key, value in inverted_index.items():
        print(key, value, file=f)
        print("-----------------------------------------------------", file=f)

# 2. RETRIEVE QUERIES
json_output = read_json("questions.json")
queries = [q["question"] for q in json_output]


# abbreviation substitution dictionary
abbr_dict = {
    '8AT': '8速自动变速器(8AT)',
    '3DHT': '3挡混合动力变速器总成(3DHT)',
    'ABS': '防抱死制动系统(ABS)',
    'A/C': '空调制冷(A/C)',
    'ACC': '自适应巡航系统(ACC)',
    'AEB': '主动式紧急刹车辅助系统(AEB)',
    'APA': '全自动泊车(APA)',
    'App': '应用程序(App)',
    'AQS': '空气质量管理系统(AQS)',
    'ARP': '防翻滚保护系统(ARP)',
    'AWD': '四驱系统(AWD)',
    'BSD': '盲点监测系统(BSD)',
    'CBC': '弯道控制系统(CBC)',
    'CMSF': '前向碰撞减缓系统术语(CMSF)',
    'CMSR': '后方碰撞预警系统(CMSR)',
    'CVW': '车辆快速接近警示系统(CVW)',
    'DBS': '紧急制动辅助系统(DBS)',
    'DOW': '开门预警系统(DOW)',
    'DPS': '驾驶员状态监测系统(DPS)',
    'D-TPMS': '主动式胎压监测系统(D-TPMS)',
    'EBD': '电子制动力分配系统(EBD)',
    'EBL': '紧急制动警示灯(EBL)',
    'E-call': '紧急救援(E-call)',
    'EDR': '事件数据记录系统(EDR)',
    'ELKA': '紧急车道保持辅助系统(ELKA)',
    'ELOW': '应急车道占用警示(ELOW)',
    'EMA': '紧急转向避让辅助系统(EMA)',
    'EPB': '电子驻车制动(EPB)',
    'ESP': '车身稳定控制系统(ESP)',
    'FCTA': '前方交叉路口预警系统(FCTA)',
    'FCW': '前方碰撞预警系统技术资料(FCW)',
    '353术语': '说明(353术语)',
    'HDC': '陡坡缓降系统(HDC)',
    'HSA': '坡道辅助系统(HSA)',
    'HUD': '抬头显示(HUD)',
    'HWA': '高速公路辅助系统(HWA)',
    'LDW': '车道偏离警示系统(LDW)',
    'LED': '发光二极管(LED)',
    'LIM': '最高限速辅助系统(LIM)',
    'LKA': '车道保持辅助系统(LKA)',
    'MAX': '最大(MAX)',
    'MIN': '最小(MIN)',
    'NFC': '近场通信(NFC)',
    'OBD': '车载诊断(OBD)',
    'PEB': '泊车紧急制动(PEB)',
    'RAB': '预警制动系统(RAB)',
    'RCTA': '后方横向来车预警系统(RCTA)',
    'RPA': '遥控泊车辅助(RPA)',
    'TCS': '牵引力控制系统术语(TCS)',
    'TLA': '交通灯提醒系统(TLA)',
    'TSI': '交通标识识别系统(TSI)',
    'USB': '通用串行总线(USB)',
    'VIN': '车辆识别代码(VIN)',
    'VPA': '语音助手(VPA)',
    'Wi-Fi': '某些类型的无线局域网技术资料(Wi-Fi)'
}
# FOR EACH QUERY
for i in range(len(queries)):
    query = queries[i]
    # query = "在什么情况下HDC会激活？"

    # query preprocess: replace abbreviations
    for key, value in abbr_dict.items():
        if key in query and value not in query:
            query = query.replace(key, value)
    print("query: ", query)
    
    
    # update the query: replace abbreviations

    seg_list_search = jieba.cut_for_search(query)  # 搜索引擎模式
    # seg_list = jieba.cut(query, cut_all=True)
    seg_list_search = [x for x in seg_list_search]
    # print("Search Mode: \t" + ", ".join(seg_list_search))  # 全模式
    
    seg_list = jieba.posseg.cut(query)
    tag_dictionary = {x.word: x.flag for x in seg_list}
    # print("Posseg Mode: \t", tag_dictionary)  # 词性标注

    keywords_jieba = []
    for word in seg_list_search:
        if word not in tag_dictionary:
            keywords_jieba.append(word)
        elif tag_dictionary[word] in ["n", "v", "a", "ad", "an", "ag", "ng", "nr", "nz", "ns", "nt", "t", "s", "i", "j", "l", "z", "vn", "vd", "vi", "vg", "f", "m", "q", "z", "b", "eng"]:
            keywords_jieba.append(word)
    # print("after list:  \t", keywords_jieba)
    keywords_jieba = set(keywords_jieba)
    # TODO method #1: generate keywords to search in the inverted index
    prompt = """你要在一本汽车说明书里查找各种问题，并且尽可能多地罗列出应该搜索的关键词，并且按优先级从大到小。此外，关键词最好为两个字，并且不容易出现在其他问题中。关键词的近义词也应该包括。
问题：什么时候应该为车辆打蜡？
答案：打蜡，涂蜡，蜡，保养，维护，时候，时间。
问题：如何启用自动泊车功能？
答案：自动泊车，自动，泊车，停车，功能，启用，开启。
问题：我如何知道我的汽车的安全带是正常工作的？
答案：安全带，安全，检查，正常工作，工作。
问题：什么是转向助力系统？
答案：转向助力系统，助力系统，转向系统，转向，助力，系统。
问题：涉水行驶前应注意什么？
答案：涉水行驶，涉水，水，行驶。
问题：在更换雨刮片时需要注意什么？
答案：更换雨刮片，雨刮片，雨刮，雨刮器，雨刷，雨。
问题：{query_str}
答案：""".format(query_str=query)
    response = pipe(prompt, max_new_tokens=64)
    response = response[0]["generated_text"][len(prompt):]
    keywords = response.split("。")[0].split("，")
    key_words_llama = set()
    keywords = set(keywords) # remove duplicates

    keywords = keywords.union(keywords_jieba)

    print(keywords)
    to_search = {}
    for word in keywords:
        # print(word, "in the following chunks: ", inverte)
        if word in inverted_index:  # will not work for longer words not in the inverted index
            for index in inverted_index[word]:
                if index not in to_search:
                    to_search[index] = 0
                # to_search[index] += 1
                to_search[index] += len(word) # give more weight to longer words
        else:
            print("word %s not in inverted index" % word, file=sys.stderr, flush=True)
            # search all the chunks
            for index in range(len(txt_list)):
                if word in txt_list[index]:
                    if index not in to_search:
                        to_search[index] = 0
                    to_search[index] += len(word)
            print("finished searching all the chunks", file=sys.stderr, flush=True)


    # sort the keys by values
    print(to_search)
    to_search = [[k,v] for k, v in sorted(to_search.items(), key=lambda item: item[1], reverse=True)]
    # to_search.sort(key=lambda x: x[1], reverse=True)
    print(to_search)
    # raise Exception("break")
    # concatenate the text chunks
    txt = ""
    if len(to_search) == 0:
        continue


    # safely ignore the middle chunks
    # for index in range(NUM_CHUNKS):
    #     i = 1
    #     if index + i + 1 >= len(to_search):
    #         break
        
    #     while to_search[index][1] == to_search[index + i][1] and to_search[index + i][1] == to_search[index + i + 1][1]:
    #         to_search[index + i][1] = 0
    #         i += 1
    
    # sort the keys by values again
    # to_search.sort(key=lambda x: (x[1], x[0]), reverse=True)
    for index in to_search[:5]:
        txt += txt_list[index[0]]
    
    with open("query_locate/%s.txt" % query.replace("/", "_"), "w", encoding="utf-8") as f:
        print(txt, file=f)
    # raise Exception("break")
# TODO： if the string becomes too long, generate an answer for each of them and query the model again to compare which one is "MORE CONCRETE"

# 顾客问：如何启用自动泊车功能？
# 小王根据汽车说明手册回答：在中央显示屏中点击自动泊车图标。

# NOTE naive template doesn't work at all
#     naive_qa_template = """顾客问：{query_str}
# 汽车说明手册上写道：{context_str}
# 小王根据汽车说明手册回答：""".format(query_str=query, context_str=txt)

# NOTE prompt engineering with examples
# TODO: context for the examples
# TODO: chain-of-thought    # model goes crazy
    qa_template = """小王是一个专业的汽车工程师。小王对汽车说明手册了如指掌。
汽车说明手册上写道：{context_str}
顾客问：作为驾驶员，在使用RPA时我应该承担哪些责任？
小王根据汽车说明手册原话，用一句话回答：作为驾驶员，在使用RPA时你应该遵守相关法律要求，并对安全停车承担全部责任。
顾客问：我如何知道我的汽车的安全带是正常工作的?
小王根据汽车说明手册原话，用一句话回答：你可以通过插入并反方向拉动锁舌来测试其固定性，并通过快速拉动来测试其自动缩回和拉紧功能。
顾客问：车辆有哪些驾驶模式？
小王根据汽车说明手册原话，用一句话回答：车辆有四种驾驶模式：纯电模式、智能模式、和运动模式。
顾客问：{query_str}
小王根据汽车说明手册原话，用一句话回答：""".format(query_str=query, context_str=txt)
    print("query length: ", len(qa_template))
    response = pipe(qa_template, max_new_tokens=128)
    response = response[0]["generated_text"]
    print(response)
    final_response = response[len(qa_template):].split("。")[0] + "。"
    print(final_response)
    # raise Exception("break")
    # counter = 1
    # TODO Allow multiple sentences in response
    # while "\n" in final_response:
    #     final_response = final_response.replace("\n", "")
    #     final_response += response[len(qa_template):].split("。")[counter] + "。"
    # print("final response:", final_response)
    json_output[i]["answer_1"] = final_response
    json_output[i]["answer_2"] = final_response
    json_output[i]["answer_3"] = final_response
    print("query: %s, response: %s, keywords: %s, chunck_distribution: %s" % (query, final_response, str(keywords), str(to_search[:10])), file=log_file, flush=True)
    # raise Exception("break")

# save json file
with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)