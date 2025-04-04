# Bert4Rec
This is an implementation of Bert4Rec.
Preprocessing is implemented with Apache beam and
model in Tensorflow 2.
This implementation has been derived from the original work of the authors.
* paper: https://arxiv.org/pdf/1904.06690
* code: https://github.com/FeiSun/BERT4Rec

## Installation
To install the requirements you can use conda. You can then simply type 
`make install` or `make install_m2` if you are on a mac m2 or m1.



## Reference

```TeX
@inproceedings{Sun:2019:BSR:3357384.3357895,
 author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
 title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 isbn = {978-1-4503-6976-3},
 location = {Beijing, China},
 pages = {1441--1450},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3357384.3357895},
 doi = {10.1145/3357384.3357895},
 acmid = {3357895},
 publisher = {ACM},
 address = {New York, NY, USA}
} 
```