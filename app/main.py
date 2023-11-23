from transformers import AutoTokenizer, AutoModel
import json
import torch
import jieba
import jieba.analyse
import jieba.posseg
import os
import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir="../model_cache")
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir="../model_cache").cuda()
model = model.eval()

with open("questions.json", "r", encoding="utf-8") as f:
    json_output = json.load(f)
queries = [q["question"] for q in json_output]

# ------------------ Generate the first answer by keyword search ------------------
log_file = open("log_keywords.tky", "w", encoding="utf-8")
txt_list = preprocess.get_chunks_naive(600, 100)    # txt_list is a list of strings
inverted_index = {}
if os.path.exists("chunks"):
    os.system("rm -rf chunks")
os.mkdir("chunks")
# write each txt chunk to file
for i in range(len(txt_list)):
    with open("chunks/%d.txt" % i, "w", encoding="utf-8") as f:
        print(txt_list[i], file=f)

# abbreviation substitution dictionary
# FOR EACH QUERY
for i in range(len(queries)):
    query = queries[i]
    print("query: ", query)

    ### Extract keywords using jieba
    seg_list_search = jieba.cut_for_search(query)  # 搜索引擎模式
    seg_list_search = [x for x in seg_list_search]
    
    seg_list = jieba.posseg.cut(query)  # 词性标注
    tag_dictionary = {x.word: x.flag for x in seg_list}

    keywords_jieba = [] # keywords extracted by jieba
    for word in seg_list_search:
        if word not in tag_dictionary:
            keywords_jieba.append(word)
        # 只保留有用词性的词
        elif tag_dictionary[word] in ["n", "v", "a", "ad", "an", "ag", "ng", "nr", "nz", "ns", "nt", "t", "s", "i", "j", "l", "z", "vn", "vd", "vi", "vg", "f", "m", "q", "z", "b", "eng"]:
            keywords_jieba.append(word)
    keywords_jieba = set(keywords_jieba)

    ### Extract keywords using LLM
    prompt = """你要在一本汽车说明书里查找各种问题，并且尽可能多地罗列出应该搜索的关键词，并且按优先级从大到小。此外，关键词最好为三个字以上，并且不容易出现在其他问题中。关键词的近义词也应该包括。
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
    response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=1024, do_sample=False)
    keywords = response.split("。")[0].split("，")

    ### Combine keywords
    keywords = set(keywords) # remove duplicates
    keywords = keywords.union(keywords_jieba)

    ### A dictionary that maps each chunk to its weight
    to_search = {}

    for word in keywords:
        if word in inverted_index:  # will not work for longer words not in the inverted index
            for index in inverted_index[word]:
                if index not in to_search:
                    to_search[index] = 0
                to_search[index] += len(word) # give more weight to longer words
        else:
            for index in range(len(txt_list)):
                if word in txt_list[index]:
                    if index not in to_search:
                        to_search[index] = 0
                    to_search[index] += len(word)

    # sort the keys by weights  #TODO: 
    to_search = [[k,v] for k, v in sorted(to_search.items(), key=lambda item: item[1], reverse=True)]
    if not to_search:
        raise Exception("no results found")
    txt = ""
    for index in to_search[:3]:
        txt += txt_list[index[0]]

    qa_template = """你是一个车载人工智能语音助手。你对汽车说明手册了如指掌，而用户无法阅读汽车说明手册。对于接下来的用户问题，你必须根据汽车说明手册的内容，并尽可能用汽车说明手册的原话回答。你的回答必须来自于汽车说明手册，如果汽车说明手册没有提到，回答“我不知道”。你只需要回答接下来的问题，不要提及任何其他内容。
例如：
问题：作为驾驶员，在使用RPA时我应该承担哪些责任？
回答：作为驾驶员，在使用RPA时你应该遵守相关法律要求，并对安全停车承担全部责任。
问题：我如何知道我的汽车的安全带是正常工作的?
回答：你可以通过插入并反方向拉动锁舌来测试其固定性，并通过快速拉动来测试其自动缩回和拉紧功能。
问题：车辆有哪些驾驶模式？
回答：车辆有四种驾驶模式：纯电模式、智能模式、和运动模式。
汽车说明手册：
{context_str}

问题：{query_str}
回答：""".format(query_str=query, context_str=txt)
    response, history = model.chat(tokenizer, qa_template, history=[], max_new_tokens=2048, do_sample=False)

    # post-processing
    final_response = response.replace("\n", "")
    final_response = final_response.split("请注意，")[0] # prevent the model from generating the next question
    final_response = final_response.split("注意！")[0] # prevent the model from generating the next question
    final_response = final_response.split("注意：")[0] # prevent the model from generating the next question
    final_response = "。".join(final_response.split("。")[:-1]) + "。"  # get rid of incomplete sentences
    if final_response == "。":
        final_response = response[len(qa_template):].replace("\n", "").split("顾客问")[0].split("�")[0]
        
    final_response = final_response.replace(" ", "")
    final_response = final_response.replace("警告！", "")
    final_response = final_response.replace("■", "")
    final_response = final_response.replace("□", "")
    final_response = final_response.replace("说明！", "")

    json_output[i]["answer_1"] = final_response
    print("query: %s, response: %s, keywords: %s, chunck_distribution: %s" % (query, final_response, str(keywords), str(to_search[:10])), file=log_file, flush=True)

# ------------------ Generate the second answer by vector search ------------------
# Input: 
#       queries: a list of query strings
#
# Output:
#       write the answers to json_output[:]["answer_2"]
# Helper:
#       preprocess.get_chunks_naive(chunk_size, overlap_size): split the pdf, return a list of strings
#       model: THUDM/chatglm3-6b
#       tokenizer: THUDM/chatglm3-6b


for i in range(len(queries)):
    json_output[i]["answer_2"] = "lpy大佬牛逼"

# ------------------ Generate the third answer by ??? ------------------
# Input: 
#       queries: a list of query strings
#
# Output:
#       write the answers to json_output[:]["answer_2"]
# Helper:
#       preprocess.get_chunks_naive(chunk_size, overlap_size): split the pdf, return a list of strings
#       model: THUDM/chatglm3-6b
#       tokenizer: THUDM/chatglm3-6b


for i in range(len(queries)):
    json_output[i]["answer_3"] = "lpy带我飞"

# ------------------ Write answers to result.json ------------------
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)