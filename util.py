import json
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def read_json(json_file:str):
    """
    return a list of dict with keys: "question", "answer_1", "answer_2", "answer_3"
    """
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


if __name__ == "__main__":
    print(read_json("questions.json")[0])