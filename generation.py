from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

MODEL_IDS = [
    'meta-llama/llama-3-1-70b-instruct',
    'mistralai/mixtral-8x7b-instruct-v01',
    'mistralai/mistral-large'
]

ALL_SUPPORTED = [
    'bigscience/mt0-xxl',
    'codellama/codellama-34b-instruct-hf',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
    'google/flan-ul2',
    'ibm/granite-13b-chat-v2',
    'ibm/granite-13b-instruct-v2',
    'ibm/granite-20b-code-instruct',
    'ibm/granite-20b-multilingual',
    'ibm/granite-3-2b-instruct',
    'ibm/granite-3-8b-instruct',
    'ibm/granite-34b-code-instruct',
    'ibm/granite-3b-code-instruct',
    'ibm/granite-7b-lab',
    'ibm/granite-8b-code-instruct',
    'ibm/granite-8b-japanese-v2-rc',
    'ibm/granite-guardian-3-2b',
    'ibm/granite-guardian-3-8b',
    'meta-llama/llama-2-13b-chat',
    'meta-llama/llama-3-1-70b-instruct',
    'meta-llama/llama-3-1-8b-instruct',
    'meta-llama/llama-3-2-11b-vision-instruct',
    'meta-llama/llama-3-2-1b-instruct',
    'meta-llama/llama-3-2-3b-instruct',
    'meta-llama/llama-3-2-90b-vision-instruct',
    'meta-llama/llama-3-405b-instruct',
    'meta-llama/llama-3-70b-instruct',
    'meta-llama/llama-3-8b-instruct',
    'meta-llama/llama-guard-3-11b-vision',
    'meta-llama/llama3-llava-next-8b-hf',
    'mistralai/mistral-large',
    'mistralai/mixtral-8x7b-instruct-v01'
]

def get_llm_engine(project_id, api_key, url, model_id=None, max_new_tokens=400):
    model_id = model_id or 'llama-3-1-70b-instruct'
    return ModelInference(
        model_id=model_id,
        params={'max_new_tokens': max_new_tokens},
        credentials=Credentials(url=url, api_key=api_key),
        project_id=project_id
    )


def get_emb_engine(project_id, api_key, url):
    client = APIClient(Credentials(url=url, api_key=api_key))
    client.set.default_project(project_id)

    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
        EmbedParams.RETURN_OPTIONS: {
            'input_text': True
        }
    }

    return Embeddings(
        model_id='intfloat/multilingual-e5-large',
        params=embed_params,
        credentials=Credentials(url=url, api_key=api_key),
        project_id=project_id
    )