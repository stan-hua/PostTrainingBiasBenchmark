# Standard libraries
import json
import os
import requests

# Non-standard libraries
import google.generativeai as genai
import replicate
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Custom libraries
import config
from config import MODEL_INFO


################################################################################
#                                  Constants                                   #
################################################################################
# Load model information from configuration
MODEL_NAME_TO_PATH = {v: k for k, v in MODEL_INFO['model_path_to_name'].items()}


################################################################################
#                               Helper Functions                               #
################################################################################
# Function to obtain access token for APIs
def get_access_token():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.client_id}&client_secret={config.client_secret}"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(""))
    return response.json().get("access_token")

# Function to get responses from the ERNIE API
def get_ernie_res(string, temperature):
    if temperature == 0.0:
        temperature = 0.00000001
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={get_access_token()}"
    payload = json.dumps({"messages": [{"role": "user", "content": string}], 'temperature': temperature})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    res_data = json.loads(response.text)
    return res_data.get('result', '')

# Function to generate responses using OpenAI's API
def get_res_openai(string, model, temperature):
    gpt_model_mapping = {"gpt-4o-mini": "gpt-4o-mini-2024-07-18	", "gpt-4o": "gpt-4o-2024-08-06"}
    gpt_model = gpt_model_mapping[model]
    api_key = config.OPENAI_KEY
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=gpt_model,
                                              messages=[{"role": "user", "content": string}],
                                              temperature=temperature)
    return response.choices[0].message.content if response.choices[0].message.content else ValueError("Empty response from API")

# Function to generate responses using Deepinfra's API
def deepinfra_api(string, model, temperature):
    api_token = config.deepinfra_api
    top_p = 0.9 if temperature > 1e-5 else 1
    client = OpenAI(api_key=api_token, api_base="https://api.deepinfra.com/v1/openai")
    stream = client.chat.completions.create(model=MODEL_NAME_TO_PATH[model],
                                            messages=[{"role": "user", "content": string}],
                                            max_tokens=5192, temperature=temperature, top_p=top_p)
    return stream.choices[0].message.content


def replicate_api(string, model, temperature):
    input={"prompt": string, "temperature": temperature}
    if model in ["llama3-70b","llama3-8b"]:
        input["prompt_template"] = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        input["prompt"]=apply_chat_template_vllm(MODEL_NAME_TO_PATH[model],string)
    os.environ["REPLICATE_API_TOKEN"] = config.replicate_api
    res = replicate.run(MODEL_NAME_TO_PATH[model],
        input=input
    )
    res = "".join(res)
    return res


# Function to generate responses using claude's API
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def claude_api(string, model, temperature):
    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=config.claude_api,
    )

    completion = anthropic.completions.create(
        model=model,  # "claude-2", "claude-instant-1"
        max_tokens_to_sample=4000,
        temperature=temperature,
        prompt=f"{HUMAN_PROMPT} {string}{AI_PROMPT}", )

    # print(chat_completion.choices[0].message.content)
    return completion.completion


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def gemini_api(string, temperature):
    genai.configure(api_key=config.gemini_api)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(string, temperature=temperature, safety_settings=safety_setting)
    return response


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def palm_api(string, model, temperature):
    genai.configure(api_key=config.palm_api)

    model_mapping = {
        'bison-001': 'models/text-bison-001',
    }
    completion = genai.generate_text(
        model=model_mapping[model],  # models/text-bison-001
        prompt=string,
        temperature=temperature,
        # The maximum length of the response
        max_output_tokens=4000,
        safety_settings=safety_setting
    )
    return completion.result


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def zhipu_api(string, model, temperature):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=config.zhipu_api)
    if temperature == 0:
        temperature = 0.01
    else:
        temperature = 0.99
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": string},
        ],
        temperature=temperature
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def gen_online(model_name, prompt, temperature, replicate=False, deepinfra=False):
    if model_name in MODEL_INFO['wenxin_model']:
        res = get_ernie_res(prompt, temperature=temperature)
    elif model_name in MODEL_INFO['google_model']:
        if model_name == 'bison-001':
            res = palm_api(prompt, model=model_name, temperature=temperature)
        elif model_name == 'gemini-pro':
            res = gemini_api(prompt, temperature=temperature)
    elif model_name in MODEL_INFO['openai_model']:
        res = get_res_openai(prompt, model=model_name, temperature=temperature)
    elif model_name in MODEL_INFO['deepinfra_model']:
        res = deepinfra_api(prompt, model=model_name, temperature=temperature)
    elif model_name in MODEL_INFO['claude_model']:
        res = claude_api(prompt, model=model_name, temperature=temperature)
    elif model_name in MODEL_INFO['zhipu_model']:
        res = zhipu_api(prompt, model=model_name, temperature=temperature)
    elif replicate:
        res = replicate_api(prompt, model_name, temperature)
    elif deepinfra:
        res = deepinfra_api(prompt, model_name, temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return res
