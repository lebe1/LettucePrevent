# LettucePrevent 🥬✋
Real-time user-rule-based hallucination prevention framework


### Installation
Install from the repository:
```bash
pip install -e .
```


### FastAPI app

To start the application, simply run the following command in the script folder
```bash
uvicorn main:app --reload
```

## Usage
- Pip package to wrap around Huggingface model
- Huggingface class extension for example .generate(hallucination_prevention=HallucinationPreventor(regex="\d"))
- WebUI to demonstrate purpose and let user experiment
	- Enable Regex generation for non-techy user with LLM in the background

## Backend Options

| Approach               | Description                                  | Example                                        |
| ---------------------- | -------------------------------------------- | ---------------------------------------------- |
| Simple                 | Restrict token-based output only             | No specific digits/words allowed etc.          |
| Multi-line case        | Restrict sentence-based output               | Not clear enough, implement first              |
| RAG case               | Restrict output based on input               | Summary task: No numbers besides input numbers |
| Advanced python method | Store variables to count on appearances etc. | No word repeated more than three times         |

## Experiments
- RAG case only approach
- Take prompts from RAGTruth dataset
- Compare LLM output with and without framework
- Compare RAGTruth output vs. generated output with framework

