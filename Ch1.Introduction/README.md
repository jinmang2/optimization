# Introduction

#### 사람들은 언제나 최적화한다!
- 투자자들은 고수익을 추구하는 한편 리스크를 피할 포트폴리오를 구축할 방법을 찾음.
- 제조업자는 상품 설계에서 디자인과 오퍼의 최대 효용에 관심을 가짐
- 엔지니어는 그들의 모델의 성능을 최적화하기 위해 파라미터(모수)를 조정

#### 자연계도 최적화!
- 물리적 시스템은 에너지를 최소상태로 유지하려는 경향이 있음
- 독립적 화학계의 분자들은 안의 전자의 잠재 에너지가 최소가 될 때까지 반응함
- 광선은 travel time을 최소화하는 경로를 따라 이동

#### So, Optimization?
- 최적화는 decision science와 물리계 분석에서 중요한 툴
- 이 툴은 사용하기 위해 **objective** 를 정의해야함!!
- 이는 연구에서 system의 성능을 수치적으로 측량할 메져!
- objective는 이익, 시간, 잠재 에너지, quantity 혹은 quantity들의 조합 등이 될 수 있음 (single number로 표현되는 것들)
- objective는 _variable_, _unknowns_ 라 불리는 시스템의 특정 특성에 의존적
- 자, 우리의 목적은 언제나 위 objective를 최적화시킬 변수의 값을 찾는 것!
- 종종 이 변수는 제한(retricted, _constrained_)되기도 함
- 예를 들어 분자 전자 밀도의 quantities와 대출 이자율은 음수가 될 수 없다 등!

#### Modeling
- 주어진 문제의 objective, variables, constraints를 정의하는 과정을 `modeling`이라고 함
- 적합한 모델을 구축하는 것은 제일 우선적으로 해야하는 작업이며 이는 경우에 따라 제일 중요한 부분이 되기도 함
- 모델이 너무 단순하면 주어진 문제의 insight를 사용하지 못할 것이며
- 모델이 너무 복잡하면 풀기가 너무 어려울 것이다.

#### Algorithm, Computer!
- 모델을 구축한 후에 `optimization algorithm`을 `computer`의 도움을 받아 solution을 찾는다
- 범용 최적화 알고리즘은 존재하지 않고 특정한 유형의 최적화 문제에 맞춘 각 알고리즘의 집합이 존재
- 어떤 알고리즘을 적용할 지에 따라 solution을 찾는 속도가 달라진다!

#### Find Solution? Checking!
- model에 optimization algorithm을 적용한 후에 solution을 성공적으로 찾았는지 확인하는 작업을 반드시 수행해야 함
- 대부분의 경우 `optimality conditions`로 알려진 우아한 수학 표현이 존재
- 이는 variable set이 문제의 solution을 찾았는지 체크
- 만일 위 조건(optimality condition)을 만족시키지 못했다면 위 조건은 solution의 추정값을 얼마나 더 향상시켜야 하는지에 대한 정보를 제공함
- model과 data에서 solution의 sensitivity을 찾는 `sensitivity analysis`와 같은 기술을 적용시켜 모델을 개선할 수 있음
- 모델을 변경하면 최적화 문제가 새로 해결되고 프로세스가 반복된다.

#### MATHEMATICAL FORMULATION
- 수학적으로, `optimization`은 해당 변수의 제약조건 하의 함수를 최대/최소하는 것
- 아래 notation으로 위를 표현
  - $x\text{ is the vector of }variables\text{, also called }unknowns\text{ or }parameters;$
  - $f\text{ if the }objective\;function\text{, a (scalar) function of }x\text{ that we want to maximize or minizize;}$
  - $c_i\text{ are }constraint\text{ functions, which are scalar functions of }x\text{ that define certain equations and inequalities that the unknown vector }x\text{ must satisfy.}$

![optim1](https://user-images.githubusercontent.com/37775784/79321847-ec78cf00-7f46-11ea-8184-a29fc75f9c34.PNG)

위 notation을 활용, 아래 최적화 문제를 풀어봅시다!

$$\min_{x\in\mathbb{R}^n}f(x)\quad\;\text{subject to}\begin{cases}
    c_i(x)=0,& i\in\Epsilon\\
    c_i(x)\geq 0& i\in\mathcal{L}
\end{cases}\quad\cdots(1.1)$$

$\mathcal{L}$과 $\Epsilon$은 quality와 inequality constraints의 Index set.
간단한 예로 아래와 같이 생각할 수 있음

$$\min{(x_1-2)}^2+{(x_2-1)^2}\quad\;\text{subject to}\begin{cases}
    {x_1}^2-x_2 \leq 0\\
    x_1+x_2 \leq 2.
\end{cases}\quad\cdots(1.2)$$

(1.2)를 (1.1)과 같이 다시 쓰면

$$\begin{array}c
f(x)={x_1-2}^2+(x_2-1)^2,\quad x=\begin{bmatrix}x_1\\x_2\end{bmatrix},\\
c(x)=\begin{bmatrix}c_1(x)\\c_2(x)\end{bmatrix}=\begin{bmatrix}-{x_1}^2+x_2\\-x_1-x_2+2\end{bmatrix},\quad\mathcal{L}=\{1,2\},\quad\Epsilon=\emptyset.
\end{array}$$

위의 firue 1.1는 objective function의 등고선을 보여준다. $f(x)$의 점들의 집합은 상수이다. 또 figure는 _feasible region_ 를 보여주는데 이는 constraints를 만족하는 점들의 집합이다. (두 제약경계 사이의 공간) 그리고 $x^\ast$는 solution을 의미한다. inequality constraints의 _infeasible side_ 는 검게 칠해져있다.

위의 예제에서 Transformation이 최적화 문제를 표현하기 위해 필요한 것을 알 수 있음
- x1, x2를 한 x로 통합하여 labeling하여 (1.1)의 형태로 표준화시키는 것이 더 자연스럽고 편리함
- f를 종종 최대화하는 경우가 존재하지만 이는 -f를 최소화시키는 문제로 변화시켜 (1.1)과 같이 사용할 수 있음

#### Example: A Transformation Problem
자, 제조업과 운송업의 간단한 문제를 생각해봅시다.
화학 회사가 $F_1$과 $F_2$의 두 공장과 각각 12개의 소매 할인판매점 $R_1,\dots,R_{12}$을 가지고 있다고 하자. 각 공장 $F_i$는 주에 $a_i$ 톤의 화학 제품을 생산할 수 있고 $a_i$는 공장의 `capacity`라 부른다. 각 소매 할인판매점 $R_j$는 주에 상품을 $b_j$ 톤만큼 필요로 한다. 한 톤당 $F_i$에서 $R_j$로의 운임료는 $c_{ij}$이다.

문제는 **각 공장에서 소매점까지 비용을 최소로 하는 양의 상품 수를 결정** 하는 것이다. 문제의 변수는 $x_{ij}$로 이는 운송할 상품 수를 의미한다. 이를 수식으로 재정립해보자.

$$\begin{array}{rll}
\text{min}&\sum_{ij}c_{ij}x_{ij}&(1.3a)\\\\
\text{subject to}&\sum_{j=1}^{12}x_{ij}\leq a_i,\quad i=1,2&(1.3b)\\\\
&\sum_{i=1}^{2}x_{ij}\geq b_{j},\quad j=1,2,\dots,12,&(1.3c)\\\\
&x_{ij}\geq 0,\quad i=1,2,\; j=1,2,\dots,12,&(1.3d)
\end{array}$$

위 식은 objective function과 제약 조건이 전부 선형 함수이기 때문에 `linear programming problem`이라고 부른다. 실전에서 모델링하려면 생산 및 재고 비용까지 고려해야 하고 운송을 위한 거래 비용도 존재할 것이다.
(1.3a)를 위의 내용을 고려하여 subscription fee $\delta$를 고려하여 $\sum_{ij}c_{ij}\sqrt{\delta+x_{ij}}\;s.t\;\delta>0$로 쓸 수도 있다. 이는 objective function이 비선형이기 때문에 `nonlinear program`이라 부른다.

![optim2](https://user-images.githubusercontent.com/37775784/79326913-9576f800-7f4e-11ea-8bb8-77c5aba49988.PNG)

#### Continuous vs Discrete Optimization
몇몇 최적화 문제에서 변수가 정수를 가지는 경우가 있음
- 예시 생략
- 이를 `integer programming problem`이라 부름
- 몇몇 경우 `mixed integer programming problem`, `MIP`라고 짧게 부르기도 함

**discrete vs continuous**
- Integer Programming 문제는 `discrete optimization problem`임
- 일반적으로 이는 integer, binary variable뿐만 아니라 ordered set의 permutations과 같은 추상 변수 객체까지 다룸
- `discrete optimization`문제를 정의할 특징은 unknown $x$가 finite set에서 그려지는지!
- 대조적으로 `continuous optimization` 문제의 `feasible set`(논 책에서 주로 다룰 문제)은 보통 uncountably infinite, $x$가 주로 실수인 값을 다룬다.
- 보통 `Continuous optimization` 문제가 풀기 쉽다!
- 함수의 smoothness가 특정 x의 objective와 constraints 정보를 사용하여 x와 가까운 점들의 함수 동작 정보를 추론(deduce)하기 쉽기 때문
- `discrete problem`의 경우 두 점이 **가깝다** 를 어떻게 측정하는지에 따라 feasible point에서 다른 point로 이동할 때 objective와 constraint의 양상이 계속 달라진다.
- `discrete optimization` 문제에서의 feasible set들은 `nonconvexity`의 극단적인 형태를 구축한다고 이해하면 된다.
- `nonconvexity`? 두 feasible points의 convex combination이 일반적으로 feasible하지 않은 것.

**Discrete Optimization**
- 이 책에서는 위 문제를 직접적으로 다루진 않을 것
- Papadimiriou와 Steiglitz가 쓴 서적,
- Nemhauser와 Wolsey가 쓴 서적,
- Cook et al. 논문,
- 그리고 Wolsey의 논문을 추천합니다.
- 근데 중요한 건 `continuous optimization` 기술이 종종 `discrete optimization` 문제를 푸는 중요한 역할을 한다는 것을 기억하라
- 예로, 해당 책 Chapter 13에서 다룰 방법이 적용이 된다더라

#### Constrained and Unconstrained Optimization
우선 여기까지 작성
