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
# model.half()

# raise Exception("break")
pipe = pipeline("text-generation", device=device, tokenizer=tokenizer, model=model)    # Use pipeline as a high level helper
# response = pipe("小王是一个专业的汽车工程师。汽车说明手册上说“设置中央显示屏显示状态：调节中央显示屏显示模式，在中央显示屏中点击-设置-显示，进入显示设置界面。点击设置中央显示屏显示模式（日间模式、夜间模式、自动)。说明！\n您可以依据个人喜好选择自动模式：\n□ 日出到日落：白天显示日间模式，晚上显示夜间模式。\n□ 自定时段：依据设置的时间段切换显示模式。\n□ 日夜模式选择自动模式后，中央显示屏会自动切换日间模式或夜间模式。小王对汽车说明手册了如指掌。小李是一名购买汽车的顾客。小李：自动模式下，中央显示屏是如何切换日间和夜间模式的？小王：", max_new_tokens=128)[0]["generated_text"]      # Run the pipeline, response is a list
# print(type(response))
# print(response)
CONTEXT_WINDOW = 2048
NUM_OUTPUT = 256
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
    结合汽车说明手册上新的信息和之前的回答（如果汽车说明手册里没有新的信息，保留原本回答），小王给出了具体可实施的回答："""
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
