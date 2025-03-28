import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel


def test_qwen2_flashattn():
    """This is used to debug the procedure of Qwen2 FlashAttention.
    
    """
    path = "/workspace/cache/modelscope/hub/qwen/Qwen2___5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        path, 
        device_map='auto', 
        torch_dtype=torch.bfloat16, 
    )    
    inputs = [
        "我喜欢读书，你给我推荐一些书吧。", 
        "我想学习编程，你能推荐一些学习资料吗？我想要Python相关的。", 
    ]
    # Tokenize inputs and pad them to the same length
    inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    # Generate outputs
    outputs = model.generate(**inputs, max_length=50)
    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)
    