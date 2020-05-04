# Chapter 2. 비제약 최적화 기초

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
$\quad\quad \text{if }x^\ast \text{is a local minimizer and }f\text{ is continuously differentiable}$
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
