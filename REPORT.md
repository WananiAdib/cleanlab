# REPORT: Incorporate Lexical Quality Metrics

The goal of this test is to improve the detection of label errors using quality metrics as mentionned in the [task](https://lemonai.notion.site/Task-11aca0aa3fa6804d9417cd6081c3a1b5). In this document, I'll detail the steps that I took and the implementation challenges. I will also talk about how to add more to this factor.

## How to run the file

Please follow the steps in the [development](./DEVELOPMENT.md) document. I have added all the necessary dependencies to  [requirements-dev.txt](./requirements-dev.txt)

## Metrics

I defined a new module [lexical_quality](lexical_quality.py), it writes all the metrics that define the quality metric:
- **Accuracy**: It is calculated by how many correctly spelled words. Returns a value from [0, 1]
- **Grammar Quality**: It is calculated by how many grammaticals errors on the total of words. Returns a value [0, 1]
- **Readability**: It is based on Flesh Reading Ease and returns results from [0, 1]

For now, I have given the same weight to all the ones above and calculated the mean as the quality metric.

For coherence, I tried to use cosine similarity between embeddings. While this algorithm, [works well](https://aclanthology.org/2021.acl-short.134.pdf) for paragraphs using sentence embeddings. The results for coherence within sentences were very noisy, and there was at first sight no satisfying results. I couldn't find any method on the litterature  used for this use-case. I finally decided to drop it for small sentences.


## Threshold modification
The approach that I took is to modify the threshold to adjust the confidence based on quality metrics.
In the equation below, $q_i$ is the new per-example threshold. $\mu_i$ represents the lexical quality metric for each example. $t$, represents all the thresholds computed for every class where $t = [t_1, t_2,\ldots,...]$. $\alpha$ is a hyperparameter that we set, by default it is $0.1$. 
$$
q_i = \alpha \cdot ( \mu_i - t) + t
$$

In the equation above, we can observe that if the quality of the text is high and the threshold is low ($\mu_i > t$), it will increase the threshold, which means less likely to be a label issue.
Similarly, if the quality of the text is low and the threshold is high, it will decrease the threshhold, which means more likely to be a label issue.
The changes to threshhold happened in [count.py](./cleanlab/count.py#L595).

## Results and Test

Please check [cleanlearning.ipynb](./cleanlearning_example.ipynb) where I present a working example.

You can run the unit-test as following:

```bash
pytest tests/test_lexical_quality.py
```

## Further Work & Optimizations

Here I detail a few things that we can make better:
- It seems that the algorithm works best when it is only punishing bad lexical quality rather than rewarding good lexical quality. We can modify to handle that.
- For accuracy and grammar quality, we can all the words by batches first, and then calculate the speeds. This will be faster as we will not have to recheck every new instance of a word.
- Multi-core processing can be done as well.
- for lexical quality, curve the function after the mean to punish heavily texts with mistakes.



