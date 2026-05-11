import asyncio
from illama.illama import ILlama

illama = ILlama(
        model_name='Codingchild/ILlama-8b-LoRA',
        retriever_name='Codingchild/medical-bge-large-en-v1.5',
        reranker_name='Codingchild/medical-bge-reranker-large',
    )

async def main():
    # prepare illama model & prompt
    chain = illama.prepare_illama(
        generation_type='w_rag'
    )

    # generate response for user's query
    response = await illama.inference(
        chain=chain,
        query='I have a headache and a fever. What should I do?'
    )

    print(response)

asyncio.run(main())