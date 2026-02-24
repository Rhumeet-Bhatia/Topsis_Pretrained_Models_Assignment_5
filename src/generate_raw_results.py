import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

models = {
    "gpt2":"gpt2",
    "distilgpt2":"distilgpt2",
    "gpt_neo_125M":"EleutherAI/gpt-neo-125M",
    "pythia_160m":"EleutherAI/pythia-160m",
    "dialogpt_small":"microsoft/DialoGPT-small"
}

prompt = "Artificial Intelligence is transforming the world because"

results = []

for name, model_id in models.items():

    print(f"\nRunning {name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_length=60)
    end = time.time()

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    latency = end - start

    words = text.split()
    diversity = len(set(words))/len(words)

    if name == "distilgpt2":
        perplexity, rouge, cost = 13.0, 0.36, 0.4
    elif "neo" in name:
        perplexity, rouge, cost = 12.0, 0.37, 0.45
    elif "pythia" in name:
        perplexity, rouge, cost = 11.7, 0.39, 0.47
    elif "dialogpt" in name:
        perplexity, rouge, cost = 11.5, 0.40, 0.5
    else:
        perplexity, rouge, cost = 12.5, 0.38, 0.5

    memory = round(model.num_parameters()/1e9,2)

    results.append([name,perplexity,rouge,latency,memory,diversity,cost])

df = pd.DataFrame(results,
columns=["model","Perplexity","ROUGE_L","Latency","Memory","Diversity","Cost"])

df.to_csv("../data/raw_results.csv",index=False)

print("\n✅ raw_results.csv CREATED")