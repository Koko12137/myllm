import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel


def test_qwen2_cache():
    """This is used to debug the procedure of Qwen2 Cache.
    """
    random.seed(1234)
    torch.manual_seed(1234)
    
    path = "/workspace/cache/modelscope/hub/qwen/Qwen2___5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        path, 
        device_map='auto', 
        max_memory={7: "48GiB"}, 
        torch_dtype=torch.bfloat16, 
    )
    # Prepare inputs for first round
    messages = [
        [
            {'role': 'user', 'content': '我喜欢读书，你给我推荐一本哲学书吧。'},
        ], [
            {'role': 'user', 'content': '我想学习编程，你能推荐一份学习资料吗？我想要Python相关的。'},
        ]
    ]
    
    # Tokenize messages and pad them to the same length
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Generate outputs
    outputs = model.generate(
        **inputs, 
        max_length=1024, 
        use_cache=True, 
        return_dict_in_generate=True, 
    )
    
    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=False)
    decoded_outputs = [
        o.split('<|im_start|>assistant')[-1] for o in decoded_outputs
    ]
    output_attn_mask = outputs['sequences'] != tokenizer.added_tokens_encoder[tokenizer.pad_token]
    
    # Add the outputs to the messages
    for i, message in enumerate(messages):
        message.append({'role': 'assistant', 'content': decoded_outputs[i]})
    
    # Add new inputs for the second round
    inputs = [
        "\n<|im_start|>user\n有什么关于计算机基础知识的吗？<|im_end|>\n<|im_start|>assistant\n", 
        "\n<|im_start|>user\n我需要网站而不是书籍。<|im_end|>\n<|im_start|>assistant\n", 
    ]
    
    # Tokenize inputs and pad them to the same length
    inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    inputs['input_ids'] = torch.cat((outputs['sequences'], inputs['input_ids']), dim=-1)
    inputs['attention_mask'] = torch.cat((output_attn_mask, inputs['attention_mask']), dim=-1)
    
    # Generate outputs
    outputs = model.generate(
        **inputs, 
        max_length=1024, 
        use_cache=True, 
        past_key_values=outputs['past_key_values'],
        return_dict_in_generate=True, 
    )
    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
    print(decoded_outputs[0])
    print(decoded_outputs[1])
    