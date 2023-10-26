from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, SummaryIndex
from llama_index.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.llms.base import llm_completion_callback
from llama_index.prompts import PromptTemplate

from typing import Any
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
model = AutoModelForCausalLM.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
model.half()

# raise Exception("break")
pipe = pipeline("text-generation", device=device, tokenizer=tokenizer, model=model)    # Use pipeline as a high level helper
response = pipe("你要在一本汽车说明书里查找各种问题，并且尽可能多地罗列出应该搜索的关键词，并且按优先级从大到小。此外，关键词最好为两个字，并且不容易出现在其他问题中。关键词的近义词也应该包括。问题：什么时候应该为车辆打蜡？答案：打蜡，涂蜡，蜡，保养，维护。问题：如何启用自动泊车功能？：自动泊车，泊车，停车，功能。问题：我如何知道我的汽车的安全带是正常工作的?答案：安全带，安全，检查，正常工作。问题：什么是转向助力系统？答案：转向助力系统，助力系统，转向系统，转向，助力。问题：涉水行驶前应注意什么？答案：涉水行驶，涉水，水，行驶。问题：在更换雨刮片时需要注意什么？答案：更换雨刮片，雨刮片，雨刮，雨刷。问题：如何通过中央显示屏进行副驾驶员座椅设置？答案：", max_new_tokens=64)
# response = pipe("你要在一本汽车说明书里查找各种问题，并且尽可能多地罗列出应该搜索的关键词，并且按优先级从大到小。问题：什么时候应该为车辆打蜡？答案：打蜡，涂蜡，蜡，保养，维护。问题：如何启用自动泊车功能？：自动泊车，泊车，停车，功能。问题：我如何知道我的汽车的安全带是正常工作的?答案：安全带，安全，检查，正常工作。问题：什么是转向助力系统？答案：", max_new_tokens=64)
print(response[0]["generated_text"])
raise Exception("break")
# print(type(response))
# print(response)
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 128

qa_template = """
    小王是一个专业的汽车工程师。
    小王对汽车说明手册了如指掌。
    汽车说明手册上写道：{context_str}
    有顾客问小王{query_str}
    小王给出了具体可实施的回答："""
refine_template = """
    小王是一个专业的汽车工程师。
    小王对汽车说明手册了如指掌。
    汽车说明手册上写道：{context_msg}
    有顾客问小王{query_str}
    小王之前的回答是：{existing_answer}
    结合汽车说明手册上新的信息和之前的回答（如果汽车说明手册里没有和{query_str}有关的信息，保留之前回答），小王给出了具体可实施的回答："""
qa_template = PromptTemplate(qa_template)
refine_template = PromptTemplate(refine_template)


class OurLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=CONTEXT_WINDOW,
            num_output=NUM_OUTPUT,
            model_name="tky"
        )
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print("complete: ")
        print(prompt)
        print("-----------------------------------------------------")
        prompt_length = len(prompt)
        response = pipe(prompt, max_new_tokens=NUM_OUTPUT)[0]["generated_text"]
        text = response[prompt_length:]
        print(text)
        while text[0] == "\n" or text[0] == " ":
            text = text[1:]
        # some output text processing here
        text = text.split("\n")[0]
        # remove quotation marks
        text = text.replace("“", "")
        # only keep the first sentence
        text = text.split("。")[0]
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()
    
llm = OurLLM()


service_context = ServiceContext.from_defaults(
    llm=llm,
    # embed_model=tokenizer,
    context_window=CONTEXT_WINDOW,
    num_output=NUM_OUTPUT
)


# Load model directly


# get_model_size(model)
# model.to(device)


# test query
documents = SimpleDirectoryReader(input_dir="data").load_data()
index = SummaryIndex.from_documents(documents, service_context=service_context)


# index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(text_qa_template=qa_template, refine_template=refine_template)
# response = query_engine.query("远程监控系统用于什么？")
response = query_engine.query("怎样加热座椅？")

print("Final response:")
print(response)
