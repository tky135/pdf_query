from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, SummaryIndex
from llama_index.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.llms.base import llm_completion_callback
from llama_index.prompts import PromptTemplate

from typing import Any
import torch
from util import read_json, read_pdf
import jieba
import jieba.analyse


# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))
# seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式
# raise Exception("brak")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_WINDOW = 2048
NUM_OUTPUT = 128
SPLIT_LENGTH = 3600

tokenizer = AutoTokenizer.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
model = AutoModelForCausalLM.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
model.half()

pipe = pipeline("text-generation", device=device, tokenizer=tokenizer, model=model)    # Use pipeline as a high level helper

# 1. BUILD INVERTED INDEX
txt = read_pdf("data/dataset.pdf")
# split txt into overlapping chunks of SPLIT_LENGTH characters
# build inverted index for each chunk
txt_list = []
inverted_index = {}
for i in range(0, len(txt), SPLIT_LENGTH):  # TODO what sbout SPLIT_LENGTH // 2
    small_txt = txt[i:i+SPLIT_LENGTH] if i + SPLIT_LENGTH < len(txt) else txt[i:]
    # chinese_parition = jieba.cut(small_txt, cut_all=True)  # full mode
    chinese_parition = jieba.cut_for_search(small_txt)  # search engine mode
    for word in chinese_parition:
        if word not in inverted_index:
            # TODO do some filtering here
            if word.isdigit():
                continue
            inverted_index[word] = set()
        inverted_index[word].add(i // SPLIT_LENGTH)
    txt_list.append(small_txt)
# write inverted index to file
with open("inverted_index.txt", "w", encoding="utf-8") as f:
    for key, value in inverted_index.items():
        print(key, value, file=f)
        print("-----------------------------------------------------", file=f)

# 2. RETRIEVE QUERIES
json_output = read_json("questions.json")
queries = [q["question"] for q in json_output]


# FOR EACH QUERY
for i in range(len(queries)):
    # query = queries[i]
    query = "远程监控系统用于什么？"
    print(query)
    # TODO method #1: generate keywords to search in the inverted index
    prompt = """你要在一本汽车说明书里查找各种问题，并且尽可能多地罗列出应该搜索的关键词，并且按优先级从大到小。此外，关键词最好为两个字，并且不容易出现在其他问题中。关键词的近义词也应该包括。
问题：什么时候应该为车辆打蜡？
答案：打蜡，涂蜡，蜡，保养，维护。
问题：如何启用自动泊车功能？
答案：自动泊车，泊车，停车，功能。
问题：我如何知道我的汽车的安全带是正常工作的？
答案：安全带，安全，检查，正常工作。
问题：什么是转向助力系统？
答案：转向助力系统，助力系统，转向系统，转向，助力。
问题：涉水行驶前应注意什么？
答案：涉水行驶，涉水，水，行驶。
问题：在更换雨刮片时需要注意什么？
答案：更换雨刮片，雨刮片，雨刮，雨刮器，雨刷。
问题：{query_str}
答案：""".format(query_str=query)
    print(len(prompt))
    response = pipe(prompt, max_new_tokens=64)
    response = response[0]["generated_text"][len(prompt):]
    print(response)
    keywords = response.split("。")[0].split("，")
    print(keywords)

    to_search = {}
    for word in keywords:
        if word in inverted_index:
            for index in inverted_index[word]:
                if index not in to_search:
                    to_search[index] = 0
                to_search[index] += 1
    # sort the keys by values
    print(to_search)
    to_search = [k for k, v in sorted(to_search.items(), key=lambda item: item[1], reverse=True)]
    print(to_search)
    # concatenate the text chunks
    # txt = ""
    # for index in to_search[:3]:
    #     txt += txt_list[index]
    # with open("temp_data.txt", "w", encoding="utf-8") as f:
    #     print(txt, file=f)
    txt = txt_list[to_search[1]]

    qa_template = """小王是一个专业的汽车工程师。小王对汽车说明手册了如指掌。
汽车说明手册上写道：{context_str}
顾客问：作为驾驶员，在使用RPA时我应该承担哪些责任？
小王根据汽车说明手册回答：作为驾驶员，在使用RPA时你应该遵守相关法律要求，并对安全停车承担全部责任。
顾客问：如何启用自动泊车功能？
小王根据汽车说明手册回答：在中央显示屏中点击自动泊车图标。
顾客问：我如何知道我的汽车的安全带是正常工作的?
小王根据汽车说明手册回答：你可以通过插入并反方向拉动锁舌来测试其固定性，并通过快速拉动来测试其自动缩回和拉紧功能。
顾客问：{query_str}
小王根据汽车说明手册回答：""".format(query_str=query, context_str=txt)
    print("query length: ", len(qa_template))
    response = pipe(qa_template, max_new_tokens=64)
    response = response[0]["generated_text"]
    print(response)
#     refine_template = """小王是一个专业的汽车工程师。小王对汽车说明手册了如指掌。
# 汽车说明手册上写道：{context_msg}
# 有顾客问小王{query_str}
# 小王之前的回答是：{existing_answer}
# 结合汽车说明手册上新的信息和之前的回答（如果汽车说明手册里没有和有关的信息，保留之前回答），小王对于给{query_str}出了具体可实施的回答："""
#     refine_template = """小王是一个专业的汽车工程师。小王对汽车说明手册了如指掌。
# 汽车说明手册上写道：{context_msg}
# 顾客问：作为驾驶员，在使用RPA时我应该承担哪些责任？
# 小王根据汽车说明手册回答：作为驾驶员，在使用RPA时你应该遵守相关法律要求，并对安全停车承担全部责任。
# 顾客问：如何启用自动泊车功能？
# 小王根据汽车说明手册回答：在中央显示屏中点击自动泊车图标。
# 顾客问：我如何知道我的汽车的安全带是正常工作的?
# 小王根据汽车说明手册回答：你可以通过插入并反方向拉动锁舌来测试其固定性，并通过快速拉动来测试其自动缩回和拉紧功能。
# 顾客问：{query_str}
# 小王根据汽车说明手册回答："""

    # qa_template = PromptTemplate(qa_template)
    # refine_template = PromptTemplate(refine_template)


    # class OurLLM(CustomLLM):
    #     @property
    #     def metadata(self) -> LLMMetadata:
    #         return LLMMetadata(
    #             context_window=CONTEXT_WINDOW,
    #             num_output=NUM_OUTPUT,
    #             model_name="tky"
    #         )
    #     @llm_completion_callback()
    #     def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    #         print("complete: ")
    #         print(prompt)
    #         print("-----------------------------------------------------")
    #         prompt_length = len(prompt)
    #         response = pipe(prompt, max_new_tokens=NUM_OUTPUT)[0]["generated_text"]
    #         text = response[prompt_length:]
    #         print(text)
    #         while text[0] == "\n" or text[0] == " ":
    #             text = text[1:]
    #             if not text:
    #                 raise Exception("empty output")
    #         # some output text processing here
    #         text = text.split("\n")[0]
    #         # remove quotation marks
    #         text = text.replace("“", "")
    #         # only keep the first sentence
    #         text = text.split("。")[0]
    #         return CompletionResponse(text=text)

    #     @llm_completion_callback()
    #     def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
    #         raise NotImplementedError()
        
    # llm = OurLLM()


    # service_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     # embed_model=tokenizer,
    #     context_window=CONTEXT_WINDOW,
    #     num_output=NUM_OUTPUT
    # )

    # # test query
    # documents = SimpleDirectoryReader(input_files=["temp_data.txt"]).load_data()
    # index = SummaryIndex.from_documents(documents, service_context=service_context)


    # # index = VectorStoreIndex.from_documents(documents)
    # query_engine = index.as_query_engine(text_qa_template=qa_template, refine_template=refine_template)
    # # query_engine = index.as_query_engine(text_qa_template=qa_template)
    # # response = query_engine.query("远程监控系统用于什么？")
    # response = query_engine.query(query)

    # print("Final response:")
    # print(response)
    break
