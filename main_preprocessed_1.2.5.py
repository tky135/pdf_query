from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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
sections = [
        "本手册相关的重要信息",
        "敬告用户",
        "联系Lynk&Co领克",
        "事件数据记录系统",
        "远程监控系统",
        "原厂精装附件、选装装备和改装",
        "无线电设备",
        "所有权变更",
        "动力电池回收",
        "车辆报废",
        "隐私告知",
        "远程查询车辆状况",
        "安全检查",
        "车辆装载",
        "前排储物空间",
        "前排储物空间",
        "第二排储物空间",
        "后备厢储物空间",
        "折叠后排座椅",
        "使用手套箱密码保护",
        "后备厢载物",
        "遮物帘",
        "车内打开/关闭尾门",
        "车内打开/关闭尾门",
        "车外打开/关闭尾门",
        "设置尾门开启角度",
        "车辆锁止/解锁状态",
        "使用遥控钥匙解锁和闭锁",
        "使用Lynk&CoApp解锁和闭锁",
        "无钥匙进入系统",
        "车内解锁和闭锁",
        "车内打开/关闭车门",
        "车外打开/关闭车门",
        "主驾座椅迎宾",
        "开门预警系统",
        "防盗系统",
        "开启/关闭防盗系统",
        "调节驾驶员座椅",
        "调节驾驶员座椅",
        "位置记忆功能",
        "位置记忆功能",
        "方向盘介绍",
        "方向盘介绍",
        "调整方向盘",
        "调节外后视镜",
        "调节内后视镜",
        "内后视镜自动防眩目",
        "胎压监测系统",
        "前雨刮和洗涤器",
        "后雨刮和洗涤器",
        "组合仪表",
        "指示灯和警告灯",
        "指示灯和警告灯",
        "查看组合仪表信息",
        "查看组合仪表信息",
        "打开/关闭远近光灯",
        "打开/关闭远近光灯",
        "打开/关闭自动大灯",
        "智能远近光控制系统",
        "大灯随动转向功能",
        "打开/关闭后雾灯",
        "打开/关闭位置灯",
        "打开/关闭转向指示灯",
        "使用阅读灯",
        "使用超车灯",
        "使用危险警告灯",
        "调节背光亮度",
        "调节背光亮度",
        "设置车内氛围灯",
        "使用接近照明灯",
        "使用伴我回家灯",
        "欢迎灯和欢送灯",
        "安全系统",
        "安全带",
        "使用安全带",
        "安全气囊",
        "颈椎撞击保护系统",
        "打开/关闭车窗",
        "打开/关闭全景天窗",
        "调节副驾驶座椅",
        "调节副驾驶座椅",
        "前排座椅加热",
        "前排座椅通风",
        "前排座椅通风",
        "头枕",
        "方向盘加热",
        "方向盘加热",
        "儿童锁",
        "儿童座椅固定装置",
        "推荐的儿童安全座椅规格",
        "语音助手",
        "车载12V电源",
        "车载12V电源",
        "智能设备充电",
        "智能设备充电",
        "智能设备充电",
        "智能设备充电",
        "外接行车记录仪",
        "遮阳板",
        "驾驶须知",
        "点火模式",
        "通过手机APP启动车辆",
        "通过遥控钥匙启动车辆",
        "车辆熄火",
        "换挡",
        "驾驶模式",
        "方向盘助力与驾驶模式联动",
        "抬头显示",
        "能量回收系统",
        "低速行驶提示音",
        "转向助力系统",
        "制动系统",
        "车身稳定控制系统",
        "制动防抱死系统",
        "电子驻车制动（EPB）",
        "自动驻车系统",
        "坡道辅助系统",
        "陡坡缓降系统",
        "加油",
        "车辆排放",
        "涉水驾驶",
        "节能驾驶",
        "冬季驾驶",
        "斜坡驻车",
        "制动防抱死系统",
        "驾驶辅助系统",
        "驾驶辅助系统传感器",
        "驾驶辅助系统传感器",
        "超速报警",
        "最高限速辅助系统",
        "自适应巡航系统",
        "高级智能驾驶",
        "高级智能驾驶",
        "交通标志识别系统",
        "驾驶员状态监测系统",
        "前方交叉路口预警系统",
        "后方横向来车预警系统",
        "车道辅助系统",
        "车道辅助系统",
        "变道辅助系统",
        "前向碰撞减缓系统",
        "后方碰撞预警系统",
        "紧急转向避让辅助系统",
        "生命体检测系统",
        "泊车辅助传感器",
        "泊车辅助传感器",
        "泊车辅助系统",
        "泊车辅助系统",
        "360°全景影像",
        "泊车紧急制动",
        "全自动泊车",
        "遥控泊车",
        "外后视镜倒车自动调节",
        "折叠/展开外后视镜",
        "开启/关闭空调",
        "调节空调温度",
        "调节空调风量",
        "空调模式",
        "调节空调出风方向",
        "主动式座舱清洁系统",
        "香氛系统",
        "空气质量管理系统",
        "空调除霜/除雾",
        "中央显示屏",
        "设置中央显示屏显示状态",
        "设置中央显示屏显示状态",
        "应用程序",
        "多媒体",
        "车辆功能界面",
        "连接设置",
        "系统设置",
        "账户设置",
        "账户设置",
        "检查车辆网络连接状态",
        "操作行车记录仪",
        "查看行车记录仪视频",
        "行车记录仪内存卡",
        "相机",
        "Lynk&CoApp",
        "创建和删除蓝牙钥匙",
        "高压警告标签",
        "混合动力电池",
        "充电安全警告",
        "车载充电设备充电",
        "随车设备快速充电",
        "预约充电",
        "车辆供电",
        "存放车辆",
        "更换遥控钥匙电池",
        "保养和维护动力电池",
        "保养和维护低压蓄电池",
        "新车磨合",
        "更换保险丝",
        "使用诊断工具读取VIN码",
        "打开前机舱盖",
        "检查发动机机油",
        "检查制动液",
        "检查冷却液",
        "添加洗涤液",
        "更换雨刮片",
        "胎压标签",
        "保养轮胎",
        "清洁车辆",
        "保养漆面",
        "车身防腐",
        "保养内饰",
        "保养项目",
        "车辆远程升级（OTA)",
        "车辆检测",
        "处理车辆故障",
        "处理车辆故障",
        "紧急救援",
        "道路救援求助服务指导",
        "应急解锁和锁止车门",
        "应急打开尾门",
        "应急解锁充电枪",
        "牵引车辆",
        "安全背心和三角警示牌",
        "补胎套装",
        "电池电量较低",
        "车辆标识",
        "车辆参数",
        "缩略语和术语"
    ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir="model_cache")
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir="model_cache").cuda()
log_file = open("log.tky", "w", encoding="utf-8")

pipe = pipeline("text-generation", device=device, tokenizer=tokenizer, model=model)    # Use pipeline as a high level helper

txt_list = tky_preprocess.get_chunks()
inverted_index = {}
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
abbr_dict = {'8AT': '8速自动变速器（8AT）', '3DHT': '3挡混合动力变速器总成（3DHT）', 'ABS': '防抱死制动系统（ABS）', 'A/C': '空调制冷（A/C）', 'ACC': '自适应巡航系统（ACC）', 'AEB': '主动式紧急刹车辅助系统（AEB）', 'APA': '全自动泊车（APA）', 'App': '应用程序（App）', 'AQS': '空气质量管理系统（AQS）', 'ARP': '防翻滚保护系统（ARP）', 'AWD': '四驱系统（AWD）', 'BSD': '盲点监测系统（BSD）', 'CBC': '弯道控制系统（CBC）', 'CMSF': '前向碰撞减缓系统术语（CMSF）', 'CMSR': '后方碰撞预警系统（CMSR）', 'CVW': '车辆快速接近警示系统（CVW）', 'DBS': '紧急制动辅助系统（DBS）', 'DOW': '开门预警系统（DOW）', 'DPS': '驾驶员状态监测系统（DPS）', 'D-TPMS': '主动式胎压监测系统（D-TPMS）', 'EBD': '电子制动力分配系统（EBD）', 'EBL': '紧急制动警示灯（EBL）', 'E-call': '紧急救援（E-call）', 'EDR': '事件数据记录系统（EDR）', 'ELKA': '紧急车道保持辅助系统（ELKA）', 'ELOW': '应急车道占用警示（ELOW）', 'EMA': '紧急转向避让辅助系统（EMA）', 'EPB': '电子驻车制动（EPB）', 'ESP': '车身稳定控制系统（ESP）', 'FCTA': '前方交叉路口预警系统（FCTA）', 'FCW': '前方碰撞预警系统技术资料（FCW）', '353术语': '说明（353术语）', 'HDC': '陡坡缓降系统（HDC）', 'HSA': '坡道辅助系统（HSA）', 'HUD': '抬头显示（HUD）', 'HWA': '高速公路辅助系统（HWA）', 'LDW': '车道偏离警示系统（LDW）', 'LED': '发光二极管（LED）', 'LIM': '最高限速辅助系统（LIM）', 'LKA': '车道保持辅助系统（LKA）', 'MAX': '最大（MAX）', 'MIN': '最小（MIN）', 'NFC': '近场通信（NFC）', 'OBD': '车载诊断（OBD）', 'PEB': '泊车紧急制动（PEB）', 'RAB': '预警制动系统（RAB）', 'RCTA': '后方横向来车预警系统（RCTA）', 'RPA': '遥控泊车辅助（RPA）', 'TCS': '牵引力控制系统术语（TCS）', 'TLA': '交通灯提醒系统（TLA）', 'TSI': '交通标识识别系统（TSI）', 'USB': '通用串行总线（USB）', 'VIN': '车辆识别代码（VIN）', 'VPA': '语音助手（VPA）', 'Wi-Fi': '某些类型的无线局域网技术资料（Wi-Fi）'}
query_modifier = {
    "反光镜": "后视镜",
    "反光境": "后视镜"
}
keywords_special = {"通风强度", }  # for those that can be searched but not identified by jieba or llama, use a systematic way to generate this
# FOR EACH QUERY
for i in range(len(queries)):
    query = queries[i]
    for key, value in abbr_dict.items():
        if key in query and value not in query:
            query = query.replace(key, value)
    for key, value in query_modifier.items():
        if key in query and value not in query:
            query = query.replace(key, value)
    print("query: ", query)

    seg_list_search = jieba.cut_for_search(query)  # 搜索引擎模式
    # seg_list = jieba.cut(query, cut_all=True)
    seg_list_search = [x for x in seg_list_search]
    
    seg_list = jieba.posseg.cut(query)
    tag_dictionary = {x.word: x.flag for x in seg_list}

    keywords_jieba = []
    for word in seg_list_search:
        if word not in tag_dictionary:
            keywords_jieba.append(word)
        elif tag_dictionary[word] in ["n", "v", "a", "ad", "an", "ag", "ng", "nr", "nz", "ns", "nt", "t", "s", "i", "j", "l", "z", "vn", "vd", "vi", "vg", "f", "m", "q", "z", "b", "eng"]:
            keywords_jieba.append(word)
    keywords_jieba = set(keywords_jieba)
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
    
    for keyword in keywords_special:
        if keyword in query:
            keywords.add(keyword)

    to_search = {}
    # add section keywords
    for keyword in sections: 
        if keyword in query:
            keywords.add(keyword)
            # extra weight for the section title
            for j in range(len(txt_list)):
                chunk = txt_list[j]
                if chunk.startswith(keyword):
                    to_search[j] = len(keyword)


    print(keywords)
    for word in keywords:
        if word in inverted_index:  # will not work for longer words not in the inverted index
            for index in inverted_index[word]:
                if index not in to_search:
                    to_search[index] = 0
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
    to_search = [[k,v] for k, v in sorted(to_search.items(), key=lambda item: item[1], reverse=True)]
    if not to_search:
        raise Exception("no results found")
    txt = ["", "", ""]

    for index in to_search[:1]:
        txt[0] += txt_list[index[0]]
    for index in to_search[:3]:
        txt[1] += txt_list[index[0]]
    for index in to_search[1:8]:
        txt[2] += txt_list[index[0]]
    
    if not os.path.exists("query_locate"):
        os.mkdir("query_locate")
    with open("query_locate/%s.txt" % query.replace("/", "_"), "w", encoding="utf-8") as f:
        print(txt, file=f)

    for txt_idx in range(3):
        qa_template = """小王是一个专业的汽车工程师。小王对汽车说明手册了如指掌。
汽车说明手册上写道：{context_str}
顾客问：作为驾驶员，在使用RPA时我应该承担哪些责任？
小王根据汽车说明手册原话回答：作为驾驶员，在使用RPA时你应该遵守相关法律要求，并对安全停车承担全部责任。
顾客问：我如何知道我的汽车的安全带是正常工作的?
小王根据汽车说明手册原话回答：你可以通过插入并反方向拉动锁舌来测试其固定性，并通过快速拉动来测试其自动缩回和拉紧功能。
顾客问：车辆有哪些驾驶模式？
小王根据汽车说明手册原话回答：车辆有四种驾驶模式：纯电模式、智能模式、和运动模式。
顾客问：{query_str}
小王根据汽车说明手册原话回答：""".format(query_str=query, context_str=txt[txt_idx])
        print("query length: ", len(qa_template))
        response = pipe(qa_template, max_new_tokens=1024, do_sample=False)
        response = response[0]["generated_text"]
        print(response)
        final_response = response[len(qa_template):].replace("\n", "")
        final_response = final_response.split("顾客问")[0] # prevent the model from generating the next question
        final_response = "。".join(final_response.split("。")[:-1]) + "。"  # get rid of incomplete sentences
        if final_response == "。":
            final_response = response[len(qa_template):].replace("\n", "").split("顾客问")[0].split("�")[0]
            
        print(final_response)
        final_response = final_response.replace(" ", "")
        final_response = final_response.replace("警告！", "")
        final_response = final_response.replace("■", "")
        final_response = final_response.replace("□", "")
        final_response = final_response.replace("说明！", "")

        json_output[i]["answer_%d" % (txt_idx + 1)] = final_response
        print("query: %s, response: %s, keywords: %s, chunck_distribution: %s" % (query, final_response, str(keywords), str(to_search[:10])), file=log_file, flush=True)


# save json file
with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)