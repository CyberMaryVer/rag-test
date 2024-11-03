from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

def get_llm_engine(project_id, api_key, url, max_new_tokens=400):
    return ModelInference(
        model_id=ModelTypes.LLAMA_3_70B_INSTRUCT,
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