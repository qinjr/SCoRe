# Sequential Collaborative Recommender (SCoRe)
A `tensorflow` implementation of all the compared models for our WSDM 2020 paper:

[Sequential Recommendation with Dual SideNeighbor-based Collaborative Relation Modeling](https://arxiv.org)

If you have any questions, please contact the author: [Jiarui Qin](http://jiaruiqin.me).


## Abstract
> Sequential recommendation task aims to predict user preference over items in the future given user historical behaviors.
The order of user behaviors implies that there are resourceful sequential patterns embedded in the behavior history which reveal the underlying dynamics of user interests. 
Various sequential recommendation methods are proposed to model the dynamic user behaviors. However, most of the models only consider the user's own behaviors and dynamics, while ignoring the collaborative relations among users and items, i.e., similar tastes of users or analogous properties of items. Without modeling collaborative relations, those methods suffer from the lack of recommendation diversity and thus may have worse performance.
Worse still, most existing methods only consider the user-side sequence and ignore the temporal dynamics on the item side.
To tackle the problems of the current sequential recommendation models, we propose Sequential Collaborative Recommender (SCoRe) which effectively mines high-order collaborative information using cross-neighbor relation modeling and, additionally utilizes both user-side and item-side historical sequences to better capture user and item dynamics. Experiments on three real-world yet large-scale datasets demonstrate the superiority of the proposed model over strong baselines.

## Citation
```
@inproceedings{qin2020sequential,
	title={Sequential Recommendation with Dual Side Neighbor-based Collaborative Relation Modeling},
	author={Qin, Jiarui and Ren, Kan and Fang, Yuchen and Zhang, Weinan and Yu, Yong},
	booktitle={Proceedings of the Thirteenth ACM International Conference on Web Search and Data Mining (WSDM '20)},
	year={2020},
	organization={ACM}
}
```
## Dependencies
- [Tensorflow](https://www.tensorflow.org) >= 1.4
- [Python](https://www.python.org) >= 3.5
- [MongoDB](https://docs.mongodb.com) and [Pymongo](https://api.mongodb.com/python/current/)
- [numpy](https://numpy.org)
- [sklearn](https://scikit-learn.org)

## Data Preparation & Preprocessing
- We give a sample raw data in the `score-data` folder. The full raw datasets are: [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42), [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) and [CCMR](http://apex.sjtu.edu.cn/datasets/6). **Remove the first line of table head**.
- Feature Engineering:
```
python3 feateng_ccmr.py # for CCMR
python3 feateng_taobao.py # for Taobao
python3 feateng_tmall.py # for Tmall
```

- Pouring data into MongoDB:
```
python3 graph_storage.py [dataset]
```

- Generate target and history sequence data
```
python3 gen_target.py [dataset] # generate target user and item
python3 gen_history_point.py [dataset] # generate sequence data
python3 sampling.py [dataset] # sampling
```

## Train the Models
For the convience of writing the code, we categorize the models in three folds: `point_model/`, `slice_model/` and `score/`.

- To run SCoRe, model_type=['SCoRe']:
```
cd score/
python3 train_score.py [model_name][gpu][dataset]
```

- To run RRN, model_name=['RRN']:
```
cd slice_model/
python3 train_time_slice_model.py [model_name][gpu][dataset]
```

- To run other models, model_name=['GRU4Rec', 'Caser', 'DELF', 'DEEMS', 'SVD++', 'SASRec']:
```
cd point_model/
python3 train_time_point_model.py [model_name][gpu][dataset]
```