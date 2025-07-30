from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

from typing import List, Dict
import uvicorn

from jinja2 import Template
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create templates directory and write HTML if not exists
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Hallucination Prevention Tool</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f9fc;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            margin-top: 40px;
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 700px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #222;
        }
        label {
            font-weight: 600;
            margin-top: 20px;
            display: block;
        }
        textarea, input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        input[type="submit"] {
            background-color: #4f46e5;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #4338ca;
        }
        pre {
            background: #f0f4f8;
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hallucination Prevention Tool</h1>
        <form method="post">
            <label>Model name:</label>
            <input type="text" name="model" value="{{ model or '' }}">

            <label>System Prompt:</label>
            <textarea name="system_prompt" rows="3">{{ system_prompt or '' }}</textarea>

            <label>Prompt:</label>
            <textarea name="prompt" rows="3">{{ prompt or '' }}</textarea>

            <label>Comma-separated strings (e.g., "hello", "dirt"):</label>
            <input type="text" name="string_list" value="{{ string_list or '' }}">

            <label>Logit threshold (e.g., -10):</label>
            <input type="number" step="0.1" name="logit_threshold" value="{{ logit_threshold or '-10' }}">

            <label>Max new tokens:</label>
            <input type="number" name="max_new_tokens" value="{{ max_new_tokens or '100' }}">

            <label>Custom Python code (optional):</label>
            <textarea name="custom_code" rows="6">{{ custom_code or '' }}</textarea>

            <input type="submit" value="Generate">
        </form>

        {% if result %}
        <h3>💬 Generated Text:</h3>
        <pre>{{ result }}</pre>

        <h3>🔍 Token-Level Metadata:</h3>
        <pre>{{ token_info }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

with open(TEMPLATES_DIR / "index.html", "w") as f:
    f.write(INDEX_HTML)


def safe_exec(user_code: str, tokens: List[str], logits: List[float]) -> Dict[str, float]:
    """
    Executes user-provided Python code in a restricted namespace.
    """
    local_vars = {}
    safe_globals = {"__builtins__": {"float": float, "len": len, "dict": dict, "str": str, "range": range}}
    func_code = (
        "def compute_probs(tokens, logits):\n"
        + "\n".join("    " + line for line in user_code.splitlines())
    )
    exec(func_code, safe_globals, local_vars)
    return local_vars["compute_probs"](tokens, logits)


@app.post("/", response_class=HTMLResponse)
async def generate(
    request: Request,
    model: str = Form(...),
    system_prompt: str = Form(""),
    prompt: str = Form(...),
    string_list: str = Form(""),
    logit_threshold: float = Form(...),
    max_new_tokens: int = Form(...),
    custom_code: str = Form(""),
):
    

    model = model.strip()
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_instance = AutoModelForCausalLM.from_pretrained(model)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_instance.eval()

    #input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_data = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=False)
    input_data = {k: v.to(model.device) for k, v in input_data.items()}

    # Clean list of strings
    string_items = [s.strip().strip('"').strip("'") for s in string_list.split(",") if s.strip()]
    sequence_bias = {}

    for phrase in string_items:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            sequence_bias[tuple(token_ids)] = logit_threshold

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        min_length=5,
        num_beams=1,
    )

    output = model_instance.generate(
        **input_data,
        generation_config=gen_config,
        #return_dict_in_generate=True,
        #output_scores=True,
        sequence_bias=sequence_bias
    )

    generated_ids = output.sequences[0][input_data.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # token_strings = tokenizer.convert_ids_to_tokens(generated_ids)
    # token_scores = output.scores
    # token_logits = [torch.nn.functional.softmax(score[0], dim=0) for score in token_scores]
    # token_probs = [
    #     (token_strings[i], float(token_logits[i][generated_ids[i]]))
    #     for i in range(len(token_strings))
    # ]

    # Apply optional custom method
    # fancy_dict = {}
    # if custom_code.strip():
    #     try:
    #         logits_vals = [float(token_logits[i][generated_ids[i]]) for i in range(len(token_strings))]
    #         fancy_dict = safe_exec(custom_code.strip(), token_strings, logits_vals)
    #     except Exception as e:
    #         fancy_dict = {"error": str(e)}

    # token_metadata_display = "\n".join(
    #     [f"{tok}: {prob:.4f}" for tok, prob in token_probs]
    # )
    # if fancy_dict:
    #     token_metadata_display += "\n\n🧠 Custom Python Output:\n" + str(fancy_dict)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": generated_text,
        #"token_info": token_metadata_display,
        "model": model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "string_list": string_list,
        "logit_threshold": logit_threshold,
        "max_new_tokens": max_new_tokens,
        "custom_code": custom_code,
    })


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
