from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    provider="hf-inference",
    max_new_tokens=128
)

print(llm.invoke("Say hello in one sentence"))
