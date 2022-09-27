<!--
 * @Author: ZhangLei mathcoder.zl@gmail.com
 * @Date: 2021-05-22 11:06:25
 * @LastEditors: ZhangLei mathcoder.zl@gmail.com
 * @LastEditTime: 2021-05-22 17:46:48
-->


# PRML学习笔记——第十章

- [PRML学习笔记——第十章](#prml学习笔记第十章)
  - [Approximate Inference](#approximate-inference)
    - [10.1 Variational Inference](#101-variational-inference)
      - [10.1.1 Factorized distributions](#1011-factorized-distributions)

## Approximate Inference

### 10.1 Variational Inference

#### 10.1.1 Factorized distributions

在Inference的时候,核心就是在于求一个posterior:$p(\mathbf{x}|\mathbf{z})$.前面的EM算法将likelihood分解后,E-step就是用$q(\mathbf{z})$在估计这个posterior.

但是往往实际中的问题会遇到posterior intractable,这就需要approximate methods.

首先为了保证近似的$q(\mathbf{Z})$即能tractable又足够flexible,假设:
$$q(\mathbf{Z})=\prod_{i=1}^{M} q_{i}\left(\mathbf{Z}_{i}\right)$$
那么lower bound可以写成(现在只考虑$q_j(z_j)$是未知的):
$$\begin{aligned}
\mathcal{L}(q) &=\int \prod_{i} q_{i}\left\{\ln p(\mathbf{X}, \mathbf{Z})-\sum_{i} \ln q_{i}\right\} \mathrm{d} \mathbf{Z} \\
&=\int q_{j}\left\{\int \ln p(\mathbf{X}, \mathbf{Z}) \prod_{i \neq j} q_{i} \mathrm{~d} \mathbf{Z}_{i}\right\} \mathrm{d} \mathbf{Z}_{j}-\int q_{j} \ln q_{j} \mathrm{~d} \mathbf{Z}_{j}+\mathrm{const} \\
&=\int q_{j} \ln \tilde{p}\left(\mathbf{X}, \mathbf{Z}_{j}\right) \mathrm{d} \mathbf{Z}_{j}-\int q_{j} \ln q_{j} \mathrm{~d} \mathbf{Z}_{j}+\text { const }
\end{aligned}$$
其中 $\ln \widetilde{p}\left(\mathbf{X}, \mathbf{Z}_{j}\right)=\mathbb{E}_{i \neq j}[\ln p(\mathbf{X}, \mathbf{Z})]$.

当我们把非常数项看成negative KL divergence,那么要maximize lower bound就是让$q_j(\mathbf{Z}_j)$越接近$\widetilde{p}\left(\mathbf{X}, \mathbf{Z}_{j}\right)$越好.最优解为:
$$\ln q_{j}^{\star}\left(\mathbf{Z}_{j}\right)=\mathbb{E}_{i \neq j}[\ln p(\mathbf{X}, \mathbf{Z})]+\text { const. }$$
