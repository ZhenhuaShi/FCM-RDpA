# FCM-RDpA
source code for [FCM-RDpA paper](https://arxiv.org/abs/2012.00060)

FCM-RDpA (Fuzzy C-Means Clustering, Regularization, DropRule, and Powerball AdaBelief; [paper](https://arxiv.org/abs/2012.00060)|[code](https://github.com/ZhenhuaShi/FCM-RDpA)|[blog](http://blog.sciencenet.cn/blog-3418535-1260629.html)) enhances the MBGD-RDA (Mini-Batch Gradient Descent with Regularization, DropRule, and AdaBound; [paper](https://ieeexplore.ieee.org/document/8930057)|[code](https://github.com/drwuHUST/MBGD_RDA)|[blog](http://blog.sciencenet.cn/blog-3418535-1214113.html)) in the following three aspects for TSK fuzzy regression model construction.

<div align=center><img src="https://github.com/ZhenhuaShi/FCM-RDpA/blob/main/Fig1.JPG"/></div>

run [demoAA.m](https://github.com/ZhenhuaShi/FCM-RDpA/blob/main/demoAA.m) to reproduce the results on the Concrete-CS dataset of Fig.3/4/8 in the paper.

run [demoPS.m](https://github.com/ZhenhuaShi/FCM-RDpA/blob/main/demoPS.m) to reproduce the results on the Concrete-CS dataset of Fig.5 in the paper.

run [demoInit.m](https://github.com/ZhenhuaShi/FCM-RDpA/blob/main/demoInit.m) to reproduce the results on the Concrete-CS dataset of Fig.6 in the paper.

run [demoGD.m](https://github.com/ZhenhuaShi/FCM-RDpA/blob/main/demoGD.m) to reproduce the results on the Concrete-CS dataset of Fig.7 in the paper.

## Citation
```
@Article{Shi2020,
  author  = {Zhenhua Shi and Dongrui Wu and Chenfeng Guo and Changming Zhao and Yuqi Cui and Fei-Yue Wang},
  journal = {IEEE Trans. on Fuzzy Systems},
  title   = {{FCM-RDpA}: {TSK} Fuzzy Regression Model Construction Using Fuzzy C-Means Clustering, Regularization, {D}rop{R}ule, and {P}owerball {A}da{B}elief},
  year    = {2020},
  note    = {submitted},
}
@Article{Wu2020,
  author  = {Dongrui Wu and Ye Yuan and Jian Huang and Yihua Tan},
  journal = {IEEE Trans. on Fuzzy Systems},
  title   = {Optimize {TSK} Fuzzy Systems for Regression Problems: Mini-batch Gradient Descent With Regularization, {D}rop{R}ule, and {A}da{B}ound ({MBGD-RDA})},
  year    = {2020},
  number  = {5},
  pages   = {1003-1015},
  volume  = {28},
}
```
