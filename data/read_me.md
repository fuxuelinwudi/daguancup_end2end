# 代码主要思想概括

1.预训练使用的是albert的ngram mask任务来替代mlm任务，新增一个structbert里提到的word struct prediction 任务，随机打乱连续的三个词，让模型来还原这三个词。

2.微调使用到的trick，(1).PGD对抗训练，(2).UDA中的TSA，(3).自定义的模型架构，(4).EMA，(5).lookahead.


# 运行全流程


##### 1.process data


> run data/code/process_data/process_data.py 
> 运行环境 ： GPU -> 单卡 RTX-3090， CPU -> inter 10700K


##### 2.build vocab


> run data/code/build_vocab/build_vocab.py
> 运行环境 ： GPU -> 双卡 RTX-2080， CPU -> 


##### 3.pretrain


> run data/code/pretrain_code/run_pretrain.py 
> 运行环境 ： GPU -> 双卡 RTX-2080， CPU -> 


##### 4.finetune


> run data/code/finetune_code/run_classify.py 
> 运行环境 ： GPU -> 单卡 RTX-3090， CPU -> inter 10700K


##### 5.predict


> run data/code/predict_code/run_predictor.py
> 运行环境 ： GPU -> 单卡 RTX-3090， CPU -> inter 10700K



##### 6.fusion


> run data/code/fusion_code/run_fusion.py
> 运行环境 ： GPU -> 单卡 RTX-3090， CPU -> inter 10700K
