## Response Dialogue-Act Prediction based on Conversational History

### Experiments
 - Dataset
 
Switch Board Dialogue Act Corpus

 - Dataset Construction
 
 ```
 {'Utterances': [['i', 'have', 'a', 'pen'], ['i', 'know']], 'DialogueAct': ['<statement>', '<understanding>'], 'caller': 'A'}
 {'Utterances': [['ok'], ['uh', 'huh']], 'DialogueAct': ['agreement', 'uninterpretable'], 'caller': 'B'}
 ```

### Usage

- train

`python train.py --expr DAestimate --gpu <gpu id>`

- evaluate

`python evaluation.py --expr DAestimate --gpu <gpu id>`

You can change hyperparamters to edit `experiments.conf`.

