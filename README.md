# LettucePrevent 🥬✋
**A real-time hallucination prevention framework to prove new features by experiments**

![Alt Text](./visualizations/NumberLogitsProcessor.gif)


### Installation
Install all necessary packages via the `requirements.txt` from the repository:
```bash
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

Currently there are three experiments available:
1. DETECTOR_TYPE 'number'
   - NumberLogitsProcessor() tries to reject all numbers that are not mentioned in the input text
2. DETECTOR_TYPE 'tinylettuce' 
   -  TinyLettuceProcessor() tries to reject all tokens based on the hallucination score of the TinyLettuce model 
   -  Currently Work In Progress!!
3. DETECTOR_TYPE 'none'
   - Default behaviour for language model without any extensions

## Experiment Results on Number Detector

All experiments have been exectued on a NVIDIA A40.

### Hallucination reduction
- Experiment executed on the 943 unique summary prompts of the RAGTruth dataset 
- Total number of text including one or several hallucinated numbers

| Model                    | Plain run | NumberLogitsProcessor run |
| ------------------------ | --------- | ------------------------- |
| Mistral 7B Instruct v0.2 | 116       | 7                         |
| Llama 7 2B               | 41        | 4                         |

### Runtime

| Model               | Plain run [s] | NumberLogitsProcessor run [s] |
| ------------------- | ------------- | ----------------------------- |
| Mistral 7B Instruct | 7630.61       | 8322.47                       |
| Llama 7 2B          | 10093.12      | 10672.77                      |

### Average runtime per generated answer
- Runtime divided by 943 

| Model               | Plain run [s] | NumberLogitsProcessor run [s] |
| ------------------- | ------------- | ----------------------------- |
| Mistral 7B Instruct | 8.09          | 8.82                          |
| Llama 7 2B          | 10.70         | 11.31                         |

