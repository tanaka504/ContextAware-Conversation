## Response Dialogue-Act Prediction based on Conversational History

### 実験設定
 - データセット
 
Switch Board Dialogue Act Corpus を使用

 - データ形式
 
 1行に「発話のリスト」，「対話行為のリスト」，「話者」の情報を含んだjson形式，
 含まれている情報は話者交替単位の発話と対話行為，及び発話者ID
 
 ```
 {'Utterances': [['i', 'have', 'a', 'pen'], ['i', 'know']], 'DialogueAct': ['<statement>', '<understanding>'], 'caller': 'A'}
 {'Utterances': [['ok'], ['uh', 'huh']], 'DialogueAct': ['agreement', 'uninterpretable'], 'caller': 'B'}
 ```
 
 - データ整形
 
 swdaのデータと前処理スクリプトをダウンロード
 
 `git clone https://github.com/cgpotts/swda.git`

 swdaプロジェクト下に`modify.py`を配置，`swda.zip`を解凍する．（swdaプロジェクト下にswdaディレクトリができる）

 `modify.py`を実行．（python2系で実行）

 `python modify.py`
 
 swdaディレクトリをResponseDApredictionプロジェクト下の`data/`ディレクトリ下にコピー
 
 `preprocess.py`を実行（python3系で実行）
 
 `python preprocess.py`
 
 `data/corpus/`ディレクトリに整形済みのデータが作成される．
 

### 実行方法
`python train.py --expr <experiment> --gpu <gpu id>`

で訓練開始，

`python evaluation.py --expr <experiment> --gpu <gpu id>`

で評価できます．

`<experiment>` は実験設定で，`experiments.conf`ファイルでパラメータの変更ができます．

#### パラメータの説明

 - `use_da`: 対話行為を使用して学習を行うか否か
 - `use_utt`: 発話を使用して学習を行うか否か
 - `use_uttcontext`: 発話履歴を使用して学習を行うか否か
 - `turn`: 話者交替をGivenにするか否か
 - `multi_dialogue`: 話者交替を1発話ごとに行うか否か
 - `state`: 使用する履歴の幅を固定するか否か


