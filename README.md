# LettucePrevent 🥬✋
This is a real-time hallucination prevention framework created to execute experiments.


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
   -  TinyLettuceProcessor() tries to rejext all tokens based on the hallucination score of the TinyLettuce model
3. DETECTOR_TYPE 'none'
   - Default behaviour for language model without any extensions


