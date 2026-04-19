General theory for our Orthogonal Collocation on Finite Elements (OCFE) framework, used in the Racelines project.

## 1. Overview

We implement a generalized segmented OCFE framework that supports any number of mesh intervals with any spacing method, making it suitable for various advanced trajectory optimization methods like $hp$-adaptive refinement. Our method is based off of \[1] and \[2].

## 2. Lagrange Polynomials

We utilize Lagrange/Barycentric interpolation polynomials between our collocation points. It is "orthonormal" in a sense: given a set of points $x_{0},\dots,x_{n},$ we define the polynomial $\ell_{i}$ as satisfying $\ell_{i}(x_{j})=\delta_{ij}$. The interpolating polynomial $p(x)$ for a function $f(x)$ is then defined as the linear combination

$$p(x)=\sum_{i=0}^n f(x_{i})\ell_{i}(x).$$

Typically, each $\ell_{i}$ is written as

$$
\ell_{i}(x)=\prod_{j=0,j\ne i}^n \frac{x-x_j}{x_{i}-x_{j}}.
$$

However, this is never used in practice as it is slow and inefficient ($O(n^2)$ creation, $O(n^2)$ evaluation), so we compute it the Barycentric method ($O(n^2)$ creation, $O(n)$ evaluation), where we precompute weights

$$
w_{i}= \left( \prod_{j\ne i} x_{i}-x_{j} \right)^{-1},
$$

yielding

$$
\ell_{i}(x)=(x-x_{j})w_{i},
$$

an alternate formula for each of the basis polynomials.

## 3. Time Segmentation

The track is divided into $K$ mesh intervals, separated by the points $t_{0},t_{1},\dots,t_{K}$. Each mesh interval has its own polynomials describing quantities of interest. We use $\tau \in [-1,1]$ to parameterize inside each mesh interval. In particular, for the interval $[t_{k-1}, t_{k}]$, we define

$$
\begin{align}
\text{Global time: }t=\overbrace{ \frac{t_k-t_{k-1}}{2} }^{ \text{Norm Factor} }\tau+\overbrace{ \frac{t_{k-1}+t_k}{2} }^{ \text{Shift }(\tau(0)) } \\
\text{Local (segment-normalized) time: }\tau=\frac{2}{t_k-t_{k-1}}t-\frac{t_{k-1}+t_k}{t_k-t_{k-1}}
\end{align}
$$

Differentiating, we have

$$
\frac{d\tau}{dt} = \frac{2}{t_{k}-t_{k-1}}
$$

which plays an important role when applying our differentiation matrices, allowing us to compute derivatives of quantities of interest using the specific polynomials for each interval with respect to a shared global parameter $t$. 

## 4. Barycentric Differentiation Matrices

Fix $N_{k}$ be the number of collocation points in an interval $k$ and $n_{q}$ be the number of state variables. We define a matrix $\mathbf{Q} \in \mathbb{R}^{(N_{k} + 2)\times n_{q}}$ where ii

$$
\text{row}_{i}(\mathbf{Q})=\mathbf{q}(\tau_{i}) \text{ i.e. } Q_{ij}=q_{j}(\tau_{i}),\quad i=1,\dots,N_{k}.
$$

Fix $p_{j}(\tau)$ be the interpolated polynomial for state component $\mathbf{q}_{j}$. With our definition of the state matrix $\mathbf{Q}$, we can write

$$
p_{j}(\tau)= \sum_{i=1}^{N_{k}} q_{j}(\tau_{i}) \ell_{i}(\tau)=\sum_{i=1}^{N_{k}}Q_{ij} \ell_{i}(\tau).
$$

Differentiating yields

$$
\dot{p}_{j}(\tau)= \sum_{i=1}^{N_{k}}Q_{ij} \dot{\ell}_{i}(\tau).
$$

We can then define the differentiation matrix $\mathbf{D} \in \mathbb{R}^{(N_{k} + 2)\times (N_{k} + 2)}$  as

$$
\mathbf{D}_{ij}=\dot{\ell}_{j}(\tau_{i}),
$$

so

$$
\frac{dQ_{ij}}{d\tau}(\tau_{l})=\dot{p}_{j}(\tau_{l})=\sum_{i=1}^{N_{k}}Q_{ij} D_{li}.
$$

Therefore, 

$$
\frac{d\mathbf{Q}}{d\tau}=\mathbf{DQ}\qquad \dot{\mathbf{Q}}=\frac{d\mathbf{Q}}{d\tau}\cdot\frac{d\tau}{dt}=\frac{2}{t_{k}-t_{k-1}} \mathbf{DQ}.
$$

#### Calculation of Barycentric Differentiation Matrix
We use the following barycentric formula to calculate the matrix $\mathbf{D}$:

$$
\begin{align}
i\ne j: & \ D_{ij}=\ell'_{j}(t_{i})= \frac{w_{j}}{w_{i}(x_{i}-x_{j})} \\
i=j: & \ D_{ii}=\ell_{i}'(t_{i})= -\sum_{j\ne i} D_{ji}
\end{align}
$$

## 5. Framework

Our OCFE formulation supports any second order system of the form

$$
\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},\mathbf{u},t)
$$

and is built with the CasADi Opti framework.
#### Segment Points
Within each mesh segment, we use the Legendre-Gauss collocation points with the endpoints -1, 1 attached for each mesh segment. Our structure is that of a "pseudo-LGR" formulation, where for each segment, dynamics are enforced at all LG points and the $-1$ endpoint.

> WIP: Currently testing LGR points + $1$ appended, behavior seems bad

This means, for $N_{k}$ collocation points in a segment $k$, we have $N_{k}+2$ optimizer variables. For each segment, we store the state matrix $\mathbf{Q}$ along with the derivative matrices $\dot{\mathbf{Q}},\ddot{\mathbf{Q}}$ as explicit optimizer variables for solver sparsity.

> We have empirically tested explicit sparse derivative variables, and they performed better than simply using the dense calculated variables for constraints.

Our derivative matrices are constrained as follows:

$$
t_{norm}=\frac{t_k-t_{k-1}}{2},\quad \text{optimizer is subject to: } t_{norm}\dot{\mathbf{Q}}=\mathbf{DQ}.
$$

#### Continuity Enforcement
We enforce continuity (and periodicity) through explicit variable sharing between segments. That is, the same optimizer variable is used for $\tau=1$ of segment $k-1$ and $\tau=-1$ of segment $k$. This explicit variable coupling provides a "stronger" constraint to the optimizer, and generally helps it stay in the feasible region.
