# Chapter 2. 비제약 최적화 기초
- 하나 알아야할게...
- ch 1장과 2장은 기초 중에 기초...
- 여기서 헤매면 안됩니다... 미적분학과 선형대수학 다시 듣고 오세요...
- 전 복습 겸 모르는 것들 생길 때마다 찾아보고 오겠습니다...

---
## 들어가기 전에...
- [smooth function](https://ko.wikipedia.org/wiki/%EB%A7%A4%EB%81%84%EB%9F%AC%EC%9A%B4_%ED%95%A8%EC%88%98)
해석학에서 smooth function은 무한 번 미분이 가능한 함수.
만일 smooth function과 모든 점에서의 taylor series 값과 함숫값이 같을 경우엔 analytic function이 된다.
  - analytic function: 국소적으로 수렴하는 멱급수로 나타낼 수 있는 함
  - 함수가 한 점에서 해석적이라는 것은 $x_0$ 근방에서 수렴하는 급수가 존재한다는 것.
  - real-analytic function은 smooth function.

- [Gradient, Jacobian, Hessian, Laplacian](https://darkpgmr.tistory.com/132)
  - gradient: $\nabla f(p)=\begin{bmatrix}\cfrac{\partial f}{\partial x_1}(p)\\\vdots\\\cfrac{\partial f}{\partial x_n}(p)\end{bmatrix}$
    - $f:\mathbb{R}^n\rightarrow\mathbb{R}$
    - $\nabla f:\mathbb{R}^n\rightarrow\mathbb{R}^n$
    - $f$의 값이 가장 가파르게 증가하는 방향
    - 함수를 지역적으로 linear approximation하거나 gradient descent로 함수의 극점을 찾는 용도로 사용.
    - By `First order Taylor Expansion`,
      $$f(x)\simeq f(p)+\nabla f(p)(x-p)$$
    - 영상처리에서 gradient는 영상의 edge 및 edge의 방향을 찾는 용도로 활용될 수 있음.
  - Jacobian: $J=\begin{bmatrix}\cfrac{\partial f}{\partial x_1}&\cdots&\cfrac{\partial f}{\partial x_n}\end{bmatrix}=\begin{bmatrix}\cfrac{\partial f_1}{\partial x_1}&\cdots&\cfrac{\partial f_1}{\partial x_n}\\\vdots&\ddots&\vdots\\\cfrac{\partial f_m}{\partial x_1}&\cdots&\cfrac{\partial f_m}{\partial x_n}\end{bmatrix}$
    - $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$
    - notation; $Df,\;J_f,\;\nabla f,\;\cfrac{\partial(f_1,\dots,f_m)}{\partial(x_1,\dots,x_m)}\in\mathbb{R}^{n \times m}$
    - 다변수 벡터 함수에 대한 일차 미분
    - By `First order Taylor Expansion`,
      $$F(x)\simeq F(p)+J_F(p)(x-p)$$
  - Hessian: $H(f)=\begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2}&\frac{\partial^2 f}{\partial x_1 \partial x_2}&\cdots&\frac{\partial^2 f}{\partial x_1 \partial x_n}\\
    \frac{\partial^2 f}{\partial x_2 \partial x_1}&\frac{\partial^2 f}{\partial x_2^2}&\cdots&\vdots\\
    \vdots&\vdots&\ddots&\vdots\\
    \frac{\partial^2 f}{\partial x_n \partial x_1}&\dots&\dots&\frac{\partial^2 f}{\partial x_2^n}
    \end{bmatrix}$
      - $\nabla f$에 대한 Jacobian Matrix
      - $f$의 이계도함수가 연속이면 혼합 편미분은 같고 Hessian은 symmetric이다.
      - 함수의 곡률(curvature) 특성을 나타내는 행렬
      - By `Second-order Taylor Expansion`,
        $$f(x)\simeq f(p) + \nabla f(p)(x-p) + \frac{1}{2}(x-p)^T H(x)(x-p)$$
      - critical point(임계점)의 종류를 판별하는데 사용
        - first-derivative가 0이 되는 지점. (stationary point)
        - local minima / local maxima / saddle point
      - 함수를 최적화시키기 위해 극점을 찾기 위해서는?
        - gradient가 0이 되는 지점(critical point)를 찾는다.
        - 이게 local minima인지 maxima인지 혹은 saddle point인지를 판별하기 위해 Hessian을 사용
      - 위를 판별하기 위한 방법이 Definite!(eigenvalue의 부호)
      - Hessian은 symmetric(연속인 함수에 대해)이므로 항상 eigenvalue decomposition이 가능.
        - 즉, 서로 orthogonal한 n개의 eigenvector를 가짐.
      - 이미지에서의 Hessian은 이미지 I(x,y)를 (x,y)에서의 픽셀의 밝기를 나타내는 함수로 봄
      - 좀 더 수학적으로 이해해보자.
      - $\text{h}\in\mathbb{R}^n$에 대해
      $$\nabla f := f(x_0+h)-f(x_0) \approx J(x_0)\text{h}+\frac{1}{2}\text{h}^T H(f)(x_0)(\text{h})$$
        - 만일 $x_0$가 임계점이면 $Df(x_0)=0$이므로
        $$\nabla f \approx \frac{1}{2}\text{h}^T H(f)(x_0)(\text{h})$$
      - $\nabla^2 f$가 연속일 경우 Hessian Matrix는 symmetric.
      - By `Spectrum Theorem`, Hessian 행렬을 OrthoDiagonal하게 만들 수 있음.
      $$\mathcal{Q}(\text{h})=\text{h}^T H(f) \text{h} = \text{h}^T Q \Lambda Q^T \text{h}=(\text{h}Q^T)^T \Lambda Q^T \text{h}$$
        - 행렬의 정부호성, definite를 체크하기 위해 아래 수식과 0을 비교
        $$\forall \text{h} \neq \overrightarrow{0},\;\text{h}^T H \text{h}=Q(\text{h})$$
        - spectral theory는 아래 수식을 의미
        $$H(f)=Q\Lambda Q^T$$
        - $\text{u}=Q^T\text{h}$로 두어 아래와 같이 수식화.
        $$\mathcal{Q}(\text{u})=\lambda_1 u_1^2 + \lambda_2 u_2^2 + \cdots + \lambda_n u_n^2$$
      - 위의 판별은 아래와 같이 진행.
        - $\Lambda$의 대각성분, 즉 Hessian 행렬의 고윳값이 모두 양수일 경우, $H(f)$는 `positive definite`이고 critical point는 `local minima`.
        - $\Lambda$의 대각성분, 즉 Hessian 행렬의 고윳값이 모두 음수일 경우, $H(f)$는 `negative definite`이고 critical point는 `local maxima`.
        - $\Lambda$의 대각성분, 즉 Hessian 행렬의 고윳값에 양수와 음수가 섞여 있는 경우, $H(f)$는 `indefinite`이고 critical point는 `saddle point`.

- [Spectrum Theorem](https://ko.wikipedia.org/wiki/스펙트럼_정리)
  - Linear Transformation을 eigenvalue와 eigenvalue의 일반화인 spectrum으로 나타내는 일련의 정리
  - for `Matrix`
    - Let $A:\mathbb{C}^n\rightarrow\mathbb{C}^n$ be Hermitian matrix.
      즉, $A=A^\ast$
      - By Spectrum Theorem, $\exist\;orthonormal\;basis\;of\;\mathbb{C}^n\;\text{constructed by }A's\;eigenvectors.\;so\;that;$
      $$A=U\text{diag}(\lambda_1,\dots,\lambda_n)U^\ast$$
        - $U:\;Unitary\;Matrix\;s.t\;U^\ast=U^{-1}$
        - $U$의 행과 열 벡터들은 $\mathbb{C}^n$의 정규 직교 기저를 이룬다.
    - Let $A:\mathbb{R}^n\rightarrow\mathbb{R}^n$ be symmetric.
      즉, $A=A^T$
      - By Spectrum Theorem, $\exist\;orthonormal\;basis\;of\;\mathbb{R}^n\;\text{constructed by }A's\;eigenvectors.\;so\;that;$
      $$A=Q\text{diag}(\lambda_1,\dots,\lambda_n)Q^\ast$$
        - $Q:\;Orthogonal\;Matrix\;s.t\;Q^T=Q^{-1}$
        - $Q$의 행과 열 벡터들은 $\mathbb{R}^n$의 정규 직교 기저를 이룬다.

- http://matrix.skku.ac.kr/2014-Album/Quadratic-form/4.Hessian%20matrix.htm
- https://ko.wikipedia.org/wiki/안장점


---

**unconstrained optimization;**
- 변수의 모든 값에 대해 어떠한 제약도 없는 실숫값 변수에 종속적인 목적 함수를 최소화
$$\min_{x}\;f(x)\quad\cdots\quad(2.1)$$
$\quad\text{where}$
$$\begin{array}{ll}
x\in\mathbb{R}^n\; s.t \; n \geq 1 \\
f:\;\mathbb{R}^n \rightarrow \mathbb{R}, \text{ smooth function}
\end{array}$$

최적화 문제에서 보통 $f$에 대한 모든 부분을 알 수 없다.
- 인도인 코끼리 문제

보통은 $f$의 함숫값 혹은 특정 점들의 집합의 미분값만 알 뿐이지만, 최적화에ㅔ서 사용할 알고리즘은 이러한 점들을 고르고 이를 통해 computing source를 최소한 줄이며 솔루션을 찾는다. 종종 $f$에 대한 정보를 얻는 것 자체가 cost가 비싼 경우가 있어 불필요한 정보를 요하지 않는 알고리즘을 선호한다.

---
#### Example 2.1
![optim_fig_2 1](https://user-images.githubusercontent.com/37775784/80961005-f7879680-8e44-11ea-904e-df95fe0505a2.PNG)

위 그림의 점을 적합해봅시다! 위 그림은 시간 $t_1,t_2,\dots,t_m$에서의 신호 $y_1,y_2,\dots,y_m$의 측량값을 기록한 그림이다. 뭐, 대략 아래의 함수로 모델링했다고 해보자(지수함수와 cos 오실레이션 함수로 추론(deduct)!).
$$\phi(t;x)=x_1 + x_2 e^{-{(x_3-t)}^2/4}+x_5\text{cos}(x_6t)$$

unknowns $x={(x_1,x_2,\dots,x_6)}^T$는 model의 모수. 우리가 원하는 건 $\phi(t_j;x)$가 관찰값 $y_j$와 최대한 가깝게 만드는 것이죠? 자 그럼 그 둘의 차이, 잔차(residual)를 아래와 같이 정의할 수 있겠죠.
$$r_j(x)=y_j-\phi(t_j;x),\quad j=1,2,\dots,m\quad\cdots\quad(2.2)$$
- 위 잔차는 관측값과 모델값 사이의 불일치(discrepancy)를 잼.

모수 $x$에 대한 추정값을 아래의 문제를 풀어서 얻을 수 있음
$$\min_{x\in\mathbb{R}^6}f(x)=r_1^2(x)+r_2^2(x)+\cdots+r_m^2(x)\quad\cdots\quad(2.3)$$
위는 `nonlinear least-squares problem`, 비제약 최적화의 특별한 경우인데, 우리의 예제에서 $n=6$이죠? 변수의 숫자가. 그러나 만약 측량값의 수 $m$이 굉장히 크면, 예를 들어 $10^5$개의 포인트가 있으면 주어진 모수 벡터 $x$에 대한 $f(x)$의 evaluation 단계는 아주 많은 연산량이 소요될 것이다.
즉, 변수의 숫자가 작더라고 objective function에 따라 cost가 비싸질 수 있음을 암시한다.

---

위 그림 2.1에서 (2.3)의 최적 해가 근사적으로 $x^\ast=(1.1,\;0.01,\;1.2,\;1.5,\;2.0,\;1.5)$, 그 함숫값이 $f(x^\ast)=0.34$라고 가정하자. objective가 0이 아니므로(제곱의 합) 관측값과 모델 예측 간에 불일치(잔차)가 반드시 존재하고, 이는 모델이 데이터 포인트를 정확하게 재생산해낼 수 없다는 것을 의미한다. 그렇다면 어떻게 $x^\ast$가 $f$의 minimizer인지 확인할 수 있을까? **"solution"** 의 정의가 뭔데? 허허... 이것이 수학이라는 것을 잊었는가 닝겐이여.

## 2.1 What is a Solution?

$Def.\text{ global minimizer of }f$
$$\text{A point }x^\ast\text{ is a global minimizer if }f(x^\ast)\leq f(x)\;\forall x\in\mathbb{R}^n$$
- 위의 $x$의 범위는 모델러에게 흥미있는 domain이 될 수도 있음.

보통 $f$에 대해 국소적인 범위만 알기 때문에 global minimizer를 찾는 것은 사실상 불가능. 즉, $f$에 대한 전반적인 그림을 모르기 때문에 알고리즘에 의해 sampling되지 않은 특정 지역에서 함숫값이 깊은 경사를 가지는지 아닌지 모름.

때문에 대부분의 알고리즘은 _local minimizer_ 를 찾는데 집중하고 이는 점 근방의 $f$ 값의 최솟값을 찾으려 노력. 수식적으로 아래와 같이 표현 가능.

$Def.\text{ local minimizer of }f$
$$\begin{aligned}
\text{A point }x^\ast\text{ is a local minimizer if }\\
\exist\;\mathcal{N}\text{ of }x^\ast\;s.t\;f(x^\ast)\leq f(x)\;\forall x\in\mathcal{N}
\end{aligned}$$
- $\mathcal{N}$은 $x^\ast$의 neighborhood
- $\text{Recall that;}$ $x^\ast$의 neighborhood는 $x^\ast$를 포함하는 open set
- 위 정의를 만족하는 점들을 _weak local minimizer_ 라 부름
- 상수함수 $f(x)=2$는 모든 점이 weak local minimizer

$Def.\text{ strict local minimizer of }f$
$$\begin{aligned}
\text{A point }x^\ast\text{ is a strict local minimizer if }\\
\exist\;\mathcal{N}\text{ of }x^\ast\;s.t\;f(x^\ast) < f(x)\;\forall x\in\mathcal{N}\text{ with }x \neq x^\ast
\end{aligned}$$
- _strong local minimizer_ 라고도 부름
- $f(x)={(x-4)}^2$는 $x=2$에서 strict local minimizer를 가짐

$Def.\text{ Isolated local minimizer of }f$
$$\begin{aligned}
\text{A point }x^\ast\text{ is a Isolated local minimizer if }\\
\exist\;\mathcal{N}\text{ of }x^\ast\;s.t\;x^\ast\text{ is only local minimizer in }\mathcal{N}
\end{aligned}$$

아래 함수를 보자.
$$f(x)=x^4 cos(1/x) + 2x^4,\quad f(0)=0$$
뭐라 부르지,,, 2차까지 연속적으로 미분 가능한 함수(twice continuously differentiable)이고 $x^\ast=0$에서 strict local minimizer를 가지는 함수.

![optim_2 11](https://user-images.githubusercontent.com/37775784/80964201-636cfd80-8e4b-11ea-991e-0470dd93fcf8.PNG)

위에를 보면 좀 더 이해가 갈 것이다. 0 점이 유일한 strict local minimizer, 즉 isloated local minimizer인가? 근방을 어떻게 잡건 항상 이보다 작은 strict local minimizer가 존재한다.
- 음... 전체적인 그림을 모르기에 정의한 local minimizer의 한계인가?
- isloated local minimizer는 유일해야 하지만, 정의 상 근방(여러 점)을 잡기 때문에 위 같은 경우엔 유일할 수 없지
- 만일 뭐 min(f)를 취하면 0이 최소인걸 알 수 있지만, 전체 그림을 우리는 아니까 그런 생각을 하는거자나?
- 뭐든 수학적인 정의 하에서 생각하라.

strict local minimizer $\rightarrow$ isloated local minimizer는 성립하지 않지만 이 역은 성립한다.

![optim_fig_2 2](https://user-images.githubusercontent.com/37775784/80964922-a8456400-8e4c-11ea-89c3-944b4fe7481c.PNG)

위 figure는 수 많은 local minimizer가 있는 함수를 표현함. 이 함수의 경우, **"trapped"**, 해당 local minimizer에서 함정에 걸릴 수 있음. 이 예제는 결코 병적이지 않다(`This example is by no means pathological.`). 분자 구조 결정과 관련된 최적화 문제에선 수백만개의 local minima가 존재할 수 있다.

이따금 $f$에 대한 global 지식을 가지고 있는 경우가 존재하고 특정 convex function의 경우 모든 local minima가 global minima가 되기도 한다.

#### Recognizing a Local Mimimum
위의 정의에서, 점 $x^\ast$가 local minima인지 아닌지 알아낼 방법은 바로 근처에 있는 모든 점을 검사하여 주변 함숫값 중에 더 작은 점이 없는지 확인하는 방법이 있다. 그러나, 만약 $f$가 _smooth_ 라면 local minima를 판별할 더 효과적인 방법이 존재한다. $f$가 이차 연속미분이 가능한 함수라 하자. $x^\ast$가 local minima(아마 strict local minimizer가 될)를 gradient $\nabla f(x\ast)$와 Hessian $\nabla^2 f(x^\ast)$로 설명할 수 있다.

smooth function의 minimizer를 연구하기 위해 사용되는 수학적 툴은 `Taylor's Theorem`이다. 테일러 정리는 이 책의 분석의 중심이고 미적분학에서 증명된 정리이다.

$Thm\;2.1.\text{ (Taylor's Theorem)}$
$\quad\quad\text{Suppose that }f:\mathbb{R}^n\rightarrow\mathbb{R}\text{ is 'continuously differentiable' and that }p\in\mathbb{R}^n.$
$\quad\text{Then we have that}$
$$f(x+p)=f(x)+\nabla f(x+tp)^T p\quad\cdots\quad(2.4)$$
$\quad \text{for some }t\in(0,1)$

$\quad\text{Moreover if }f\text{ is 'twice continuously differentiable', we have that}$
$$\nabla f(x+p)=\nabla f(x)+\int_0^1 \nabla^2 f(x+tp) p\; dt\quad\cdots\quad(2.5)$$

$\quad\text{and that}$
$$f(x+p)=f(x)+\nabla f(x)^T p +\frac{1}{2}p^T\nabla^2 f(x+tp) p \quad\cdots\quad(2.6)$$
$\quad \text{for some }t\in(0,1)$

$Thm\;2.2.\text{ (First-Order Necessary Conditions)}$
$\quad\quad \text{if }x^\ast \text{is a }local\;minimizer\text{ and }f\text{ is }continuously\;differentiable$
$\quad \text{in an open nighborhood of }x^\ast\text{, then }\nabla f(x^\ast)=0$

$pf)$
```
(수식 쓰기 귀찮아서 말로 증명)
x가 local minima이고 f가 x의 open neighborhood 안에서 연속이고 미분 가능한 함수일 때,
x에서의 gradient가 0이 아니라고 가정하자.
p를 x에서의 미분값의 음수라 정의하면
p^T grad(f(x))는 x에서의 미분값의 제곱에 음수를 취한 값이다.
조건에서 x 근방에서 grad(f)가 연속이니, 아래 식을 만족하는 scalar 양수 T가 존재한다.
    p^T grad(f(x + tp)) < 0  for all t \in [0, T]
임의의 t^hat \in (0, T]에 대하여 테일러 정리를 사용하여 아래와 같이 쓸 수 있다.
    f(x + t_hat p) = f(x) + t_hat p^T grad(f(x + tp)),  for some t in (0, t_hat)
위 두 식에서 아래의 정보를 얻을 수 있다. (아래 식의 우항의 뒤에 텀이 음수니까)
    f(x + t_hat p) < f(x) for all t_hat \in (0, T]
어라? x가 local minima인데 근방에 이보다 작은 x + t_hat p가 존재하네?
즉, x가 local minima가 아니란 소리고, 이는 조건과 모순된다.
Hence, 해당 조건에서 x에서의 gradient는 0이다. 증명 끝.
```

$\nabla f(x^\ast)=0$을 만족하는 점 $x^\ast$를 _stationary point_ 라 부른다. 정리 2.2에 따르면 위 조건을 만족하는 local minimizer는 stationary point가 된다.

$Recall\;that;$
- $\text{matrix }B\text{ is }positive\;definite\text{ if }p^T Bp>0\;\forall p \neq 0$
- $\text{matrix }B\text{ is }positive\;semidefinite\text{ if }p^T Bp \geq 0\;\forall p \neq 0$
- 여기서 $p$는 $n$차원 실수의 non-zero 열 벡터이다.
- $B$는 $n \times n$의 정방행렬. (symmetric or Hermitian)
- 위 행렬 $p^T B p$는 scalar 값이다.
- [정부호 행렬(definite matrix)](https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B6%80%ED%98%B8_%ED%96%89%EB%A0%AC)을 참고하라.
- 이를 만족시킬 시의 특징은?
  - eigenvalue가 모두 양수
  - 임의의 두 벡터 $x, y$에 대해 $<x, y>=x^\ast M y$로 내적을 정의하는 것이 가능하다.
  - $M$은 gram 행렬. 어떠한 linear independent vector $x_1,\cdots,x_n$이 존재하여 $M_{ij}=x_i^\ast x_j$가 성립.
  - $M=LL^\ast$가 만족하는 lower triangle matrix $L$이 유일하게 존재.
    - 이러한 분해를 Cholesky Decomposition이라고 함.

$Thm\;2.3.\text{ (Second-Order Necessary Conditions)}$
$\quad\quad \text{if }x^\ast \text{is a }local\;minimizer\text{ and }f\text{ is }continuously\;differentiable$
$\quad \text{in an }open\;nighborhood\text{ of }x^\ast\text{, then }\nabla f(x^\ast)=0\text{ and }\nabla^2 f(x^\ast)\text{ is }positive\;semidefinite$

$pf)$
```
thm 2.2에 의해 grad(f(x))=0인 것은 clear.
x가 local minimizer이고 f가 continuously differentiable이지만
grad^2(f(x))가 positive semidefinite가 아니라고 가정하자.
positive semidefinite가 아니니 정의에서 p^Tgrad^2(f(x))p < 0인 p 벡터를 택하자.
조건에서 grad^2f가 x 근방에서 연속이기 때문에 아래 조건을 만족하는 양수 T가 존재한다.
    p^T grad^2(f(x + tp)) p < 0  for all t \in [0, T]
x 근방에 Taylor 급수 expansion을 통해,
    f(x + t_hat p) =
    f(x) + t_hat p^T grad(f(x)) + 1/2 t_hat^2 p^T grad^2(f(x + tp)) p < f(x)
    for all t_hat \in (0, T] and some t \in (0, t_hat)
x의 근방에서 f가 감소하는 방향이 존재하므로 x는 local minima가 아니다.
즉, contradiction 발생. grad^2(f(x))는 positive semidefinite이다.
증명 완료.
```

$Thm\;2.4.\text{ (Second-Order Sufficient Conditions)}$
$\quad\quad \text{Suppose that }\nabla^2 f\text{ is }continuous\text{ in an }open\;neighborhood\text{ of }x^\ast$
$\quad\text{and that }\nabla f(x^\ast)=0\text{ and }\nabla^2 f(x^\ast)\text{ is }positive\;definite.$
$\quad\text{Then }x^\ast\text{ is a }strict\;local\;minimizer\text{ of }f.$

$pf)$
- https://math.stackexchange.com/questions/1477978/positive-definite-and-continuous-function
- https://en.wikipedia.org/wiki/Sylvester%27s_criterion
- https://en.wikipedia.org/wiki/Minor_(linear_algebra)
- http://www.pitt.edu/~luca/ECON2001/lecture_08.pdf

$Recall\;that$
- $principal\;submatrix\text{ of }A\in\mathbb{R}^{n \times n}$: k번째 행과 열을 제거하여 얻어지는 행렬
- $principal\;minor\text{ of }A$: principal submatrix의 determinant
- $\text{leading principal submatrix of order k}\text{ of }M\in\mathbb{R}^{n \times n}$: 행렬의 $n-k$개의 행과 열을 제거하여 얻어지는 행렬
- $\text{leading principal minor of }M$: leading principal submatrix의 determinant
- `all of its leading principal minors(upper-left subdeterminants)`?
    $\begin{array}{ll}
    \text{if }M\in\mathbb{R}^{n \times n},\\
    \begin{vmatrix}M_{11}\end{vmatrix},\;\begin{vmatrix}M_{11}&M_{12}\\M_{21}&M_{22}\end{vmatrix},\;\begin{vmatrix}M_{11}&M_{12}&M_{13}\\M_{21}&M_{22}&M_{23}\\M_{31}&M_{32}&M_{33}\end{vmatrix},\;\cdots\;,\;\begin{vmatrix}M_{11}&M_{12}&\cdots&M_{1n}\\\vdots&\vdots&\ddots&\vdots\\M_{n1}&M_{n2}&\cdots&M_{nn}\end{vmatrix}
    \end{array}$
```
조건에서 Hessian(2차 미분)이 x에서 연속이고 positive definite이므로,
open ball D={z|dist(z,x)<r}의 모든 point z에 대해 grad^2(f(z))가 여전히
positive definite가 되도록하는 양수의 반지름 r을 택할 수 있다.
    (why? >>
        By Sylvester's criterion,
            symmetric(Hermitian) matrix will be positive definite iff(if and only if)
            all of its leading principal minors(upper-left subdeterminants) are positive.
        Note that; leading principal minors는 행렬의 성분에 의해 지속적으로 변화함.
        that is,
            Suppose that A is a positive definite 행렬.
            만약 A의 성분을 아주 조금 변화시킨다면, 이 principal minors는 여전히 양수일 것이다.
            (why? 연속! epsilon 반경 안에 잡을 수 있다는 얘기.)
        In other words,
            A 주변에서 항상 양수의 leading principal minors를 가지는 행렬의 open ball을 찾을 수 있다.

        자, 말로 설명하지 말고 수식적으로 보자.

        위를 수식으로 표현하면 아래와 같다.
            \exist \epsilon > 0 s.t
                if MatNorm(A - grad^2(f(x))) < \epsilon,
                then A has positive minors.
        Since grad^2(f(x))가 연속,
            \exist \delta s.t
                if dist(z - x) < \delta,
                then MatNorm(H(f(z)) - H(f(x))) < \epsilon
        즉, dist(z-x) < \delta면 grad^2(f(x))는 positive definite.
        r = \delta로 채택.
        그러면 open ball D = {z | dist(z, x) < r}의 모든 점 z에 대해 여전히
        H(f(z))가 positive definite.

        +추가) 여기서 위에 MatNorm(H(f(z)) - H(f(x))) < \eps부분, 여기까진 내가 생각했는데,
        여기서 quadratic form으로 바꿔서 그 값의 하나하나가 eps로 잡혀야하나?
        하는 이상한 생각까지 가서... 오 그래도 leading principal minors 개념으로
        뭔가 compact하게 증명했다. 맘에 들어!
    )
norm(p) < r인 nonzero vector p를 택하자.
x + p \in D에 대해,
    f(x+p) = f(x) + p^T grad(f(x)) + 0.5 p^T grad^2(f(z)) p
           = f(x) + 0.5 p^T grad^2(f(z)) p (since grad(f(x))=0.)
    where z = x + tp for some t \in (0, 1).
Since z \in D, p^T H(f(z)) p > 0. (because remain positive definite.)
Therefore f(x + p) > f(x).
Hence, x is a strict local minimizer of f.
증명 완료.
```

$Note\;that$;
- 위 정리는 충분 조건이지 필요조건이 아님.
- $f(x)=x^4$에서 $x^\ast=0$은 strict local minima지만 Hessian matrix는 소실(Vanish)되며 즉 positive definite가 아님.

objective function이 convex인 경우, local과 global minimizer는 쉽게 정의할 수 있음.

$Thm\;2.5.$
$\quad\quad\text{When }f\text{ is convex, any }local\;minimizer\text{ }x^\ast\text{ is a }global\;minimizer\text{ of }f.$
$\quad\quad\text{If in addition }f\text{ is }differentiable,\text{ then any }stationary\;point\;x^\ast\text{ is a }global\;minimizer\text{ of }f.$

$pf)$
```
Part 1.
    Suppose that x* : local but not a global minimizer.
    Then, \exist point z \in \mathbb{R}^n s.t f(z) < f(x*). (global minima가 아니니까!)
    Consider the "line segment" that joins x* to z,
        x = \lambda z + (1 - \lambda) x*,    for some \lambda \in (0, 1] (2.7)
    By the convexity property for f,
        f(x) \leq \lambda f(z) + (1 - \lambda) f(x*) < f(x*) (2.8)
        (why?
            \lambda f(z) + (1 - \lambda) f(x*) < \lambda f(x*) + (1 - \lambda) f(x*) = f(x*)
        )
    위에서 x* 근방 \mathcal{N}이 (2.7)의 "line segment"의 일부를 포함하기 때문에
    x \in \mathcal{N} s.t (2.8)한 x를 찾을 수 있고,
    Hence, x*는 local minimizer가 아니다.
증명 완료.

Part 2.
    증명 안해.
```

기초 미적분학에 기반하는 위의 결과들은 비제약 최적화 알고리즘의 기반을 제공
모든 알고리즘은 $\nabla f(\cdot)$이 소실되는 점을 찾는다.

#### Nonsmooth problems
본 교재에서는 smooth function(무한번 미분 가능)에 초점을 맞춤. 뭐, 주로 2차 미분이 존재하고 연속인 함수들. 그러나 대부분 불연속에 smooth하지 않은 경우가 존재. 일반적인 불연속 함수에서 minimizer를 정의하는 것은 불가능하지만, 함수가 smooth한 부분을 포함한다면 개별적인 smooth piece에서의 minimizer를 찾는 것은 가능.

![optim_fig_2 3](https://user-images.githubusercontent.com/37775784/81149412-2292f700-8fb9-11ea-9585-adcb34b02f16.PNG)

위의 함수는 모든 곳에서 연속이지만 특정 점에서 미분이 불가능하다.
이걸 뭐, `subgradient`, `generalized gradient`등의 nonsmooth optimizatoin 방법론으로 해결할 수 있다네?

## 2.2 Overview of Algorithms

지난 40년(언제 출판됐지?)간 smooth functions에 대한 비제약 최적화 알고리즘들이 수없이 개발됐다고 하네. 이를 Chapter 3~7장에서 소개.
모든 비제약 최적화 문제는 사용자에서 시작 지점을 요구.
API와 dataset에 대한 이해를 가진 유저는 $x_0$를 합리적인 이유로 solution을 추정하는데 좋은 position으로 채택할 것. (**제일 중요!!!**)
위와 같이 할 수도 있지만, 시스템적으로 혹은 임의로 정해서 시작할 수도 있겠지.

$x_0$에서 시작하여 최적화 알고리즘은 더 이상의 진척이 없거나 solution 지점이 충분한 정확도로 근사됐을 때 종료되는 반복 $\{x_k\}_{k=0}^{\infty}$ sequence를 생성한다. 알고리즘은 $x_k$, 이전 점들에서의 $f$에 대한 정보를 사용하여 iterate $x_k$에서 어디로 갈지를 결정한다.

$x_k$에서 $x_{k+1}$로 어떻게 update해야하는지에 대한 두 가지 기본 전략이 있다. 이 책의 대부분의 알고리즘들은 둘 중 하나의 접근 방식을 따른다.

#### Two strategies: Line search and Trust region

**Line Search strategy**
- direction $p_k$를 고르고 해당 방향을 따라 더 작은 함수값을 가지는 다음 iterate를 찾는다.
- $p_k$를 따라 갈 거리는 아래 step length $\alpha$를 찾기 위해 1차원 최소화 문제를 근사적으로 풀어서 얻어진다.
    $$\min_{\alpha>0}f(x_k + \alpha p_k)\quad\cdots\quad (2.10)$$
- (2.10)을 풀면 $p_k$ 방향으로 부터 최대 이득을 얻을 수 있지만 정확하게 최소화하는 지점을 찾아가는 것은(올곧게) 비쌀 수 있으며 보통 불필요하다.
- 대신, `line search` 알고리즘은 (2.10)의 최소치에 근접한 실험 단계를 찾을 때까지 제한된 수의 test 단계 length를 생성
- 새로운 지점에서 새로운 검색 방향과 step length를 재계산하고 이 과정을 반복.

**Trust Region strategy**
- $f$로부터 얻은 정보를 토대로 `model function` $m_k$를 구축
    - $x_k$ 근방에서의 움직임이 실제 목적 함수 $f$와 유사하게 움직이게 하는 함수
- $x$가 $x_k$에서 멀리 떨어져 있을 시 모델 $m_k$는 $f$를 잘 근사하지 못하기에 $m_k$의 최소화에 대한 search를 $x_k$ 주변의 일부 region으로 제한.
- 즉, 아래의 subproblem을 근사적으로 푸는 후보 step $p$를 찾는다.
    $$\min_p m_k(x_k+p),\quad\text{where }x_k+p\text{ lies inside the trust region.}\quad\cdots\quad(2.11)$$
- solution 후보가 $f$에서 감솟값이 시원치 않으면, trust region을 너무 크게 잡았다고 결론짓고 이를 수축시켜 (2.11)을 다시 푼다.
- 보통 trust region은 $\Vert p \Vert_2 \leq \Delta$로 정의된 ball이다.
    - 위에서 $\Delta > 0$는 trust-region의 반지름이다.
- model $m_k$는 일반적으로 아래와 같이 quadratic function으로 정의된다.
    $$m_k(x_k+p)=f_k+p^T \nabla f_k + \frac{1}{2} p^T B_k p,\quad\cdots\quad (2.12)$$
    - $f_k,\nabla f_k,B_k$: scalar, vector, matrix
    - $f_k$와 $\nabla f_k$는 $x_k$에서의 함수 및 gradient 값으로 선택되므로 $m_k$와 $f$는 현재 iterate $x_k$에서 1차까지 공유함.
    - $B_k$는 Hessian $\nabla^2 f_k$ 혹은 이를 근사한 행렬.

아래 예시를 한 번 보자.

---
$f(x)=10(x_2 - x_1^2)^2 + (1 - x_1)^2$가 objective function으로 주어졌다고 가정하자.
point $x_k=(0,1)$에서의 gradient와 Hessian은
$$\nabla f_k=\begin{bmatrix}-2\\20\end{bmatrix},\quad\nabla^2 f_k=\begin{bmatrix}-32&0\\0&20\end{bmatrix}$$

![image](https://user-images.githubusercontent.com/37775784/81159997-8838b080-8fc4-11ea-91f3-71d6a81c9c3a.png)

위 그림은 $B_k=\nabla^2 f_k$, (2.12)의 quadratic model의 contour line이다. 추가적으로 objective function $f$와 trust region도 같이 표시하고 있다. 그림은 또한 model $m_k$가 1과 12의 값을 가질 때의 contour line을 표시하고 있다. 위 그림에서 주의할 점은, 후보 iterate 실패 후 trust region의 크기를 줄일 때마다 $x_k$에서 다음 후보까지의 거리가 짧아지고 이는 통상 다음 후보와 다른 방향을 가리키게 된다는 점이다. trust-region 전략은 single search direction을 가지는 line search와 이런 점에서 다르다.

즉, `line search`와 `trust-region` 접근법은 다음 iterate로 옮길 때 사용할 `direction`과 `distance`를 고르는 방식에서 차이가 난다. line search는 고정된 direction $p_k$에서 시작하고 적절한 step length $\alpha_k$라 불리는 distance를 정의한다. trust region에서는 최대 distance(trust-region radius) $\Delta_k$를 고르고 방향을 탐색하며 이 distance constraint에서 최고의 성능을 보이는 쪽으로 발걸음을 옮긴다. 만일 해당 step이 불만족스럽다면 distance measure $\Delta_k$를 줄이고 다시 시도한다.



---
