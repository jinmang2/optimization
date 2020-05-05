> 본 문서는 ch 2장을 공부하며 spectrum theory에서의 spectrum이 뭔지
> 알기 위해 wikipedia의 내용을 번역 및 정리한 내용입니다.

# Spectrum
spectrum에 대한 일반적인 정의는,
```
특정 값 집합에 한정되지 않고 연속체(continuum) 내에서 무한히 변할 수 있는 조건 또는 값.
```

다양하게, 뭐 아래 항목들에서 언급된다고 한다.

- Science and technology
  - Physics
  - Medicine
  - Mathematics
- Arts and entertainments
  - Publications
    - Student newspapers
  - Television
  - Music
  - Other arts and entertainment
- Buildings and structures
- Organizations
- Products
- Other uses
- see also...

위에서 Science and technology에서 정의된 spectrum만 관심있고 아래 내용들에 대해 공부하고 정리한다.
- Physics/Electromagnetic spectrum; 개념만
- Physics/Discrete spectrum & Continuous spectrum; 약간 세밀하게
- Physics/Frequency spectrum/Spectrogram; 자세하게, 음악 생성 모델 및 sequential한, 주파수 데이터에 필수적인 개념.
- Physics/Mass spectrum; 개념만
- Mathematics/Spectrum; 자세하게

## Physics

### Continuous spectrum
![img](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Deuterium_lamp_1.png/360px-Deuterium_lamp_1.png)

일반적으로 각 값과 다음 값 사이에 양의 간격이 있는 수학적 의미에서 Discrete spectrum과 반대로 실수 구간으로 가장 잘 설명되는 일부 Physical quantity(에너지나 파장 등)에 대한 attainable한 값들의 집합을 의미.

예시로 자유 전자가 수소 이온에 결합되어 광자를 방출하기 때문에 발생하는 수소 원자가 방출하는 빛의 스펙트럼의 일부분이라고 한다.

Hamiltonian operator와 같이 이산 스펙트럼과 연속 스펙트럼은 functional space에 작용하는 linear operator의 spectrum decomposition에서 서로 다른 부분으로 기능 분석에서 모델링 가능하다고 한다.

### Discrete spectrum

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Hydrogen_spectrum.svg/360px-Hydrogen_spectrum.svg.png)
- 수소의 emission spectrum의 이산적인 부분

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Solar_Spectrum.png/360px-Solar_Spectrum.png)
- 대기(노란색) 위와 해수면(빨간색)에서 햇빛의 스펙트럼으로, 이산 부분(O2로 인한 선)과 연속 부분(H2O로 표시된 밴드 등)이 있는 흡수 스펙트럼을 표시

physical quantity은 한 값과 다음 값 사이에 간격을 두고 구별되는 값만 취하면 discrete spectrum을 갖는다고 함.
- quantization?

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Oh_No_Girl_Spectrogram.jpg/360px-Oh_No_Girl_Spectrogram.jpg)

위 그림은 어린 소녀가 말한 "아, 안돼!"의 `Acustic Spectrogram`이 소리(밝은 오렌지 선)의 discrete spectrum이 시간(수평축)에 따라 어떻게 변화하는지 보여준다.
- 값과 값 사이에 간격을 두고 수집, 이산적 스펙트럼!

현이 진동, 금속 구멍의 전자파, 진동하는 별의 음파, 고에너지 입자 물리학의 공명 등 많은 현상에서 보여진다고 함.

spectrum decomposition에 의해 수학적으로 모델링될 수 있다고 함.

**Origins of discrete spectra**
- Classical Mechanics
  - discrete spectra는 종종 bounded된 객체나 도메인에서 waves나 oscillations과 연관이 있고, 이는 수학적으로 시간 및 공간의 함수로 연속 변수(strain 혹은 pressure)의 변화(evolution)을 묘사하는 differential operators의 eigenvalues으로 식별 가능.
  - 일부 비선형 oscillator에 의해 생성, 이 값은 non-sinusoidal waveform을 가짐. 예시로 포유류의 성대에 의해 만들어지는 소리 등.
  - 단일 spectral line으로 구성된 궁극의 "discrete spectrum"을 가지는 `sinusoidal signal`이 non-linear filter에 의해 수정될 때 강력한 harmonics의 출현과도 관련이 있다.
    - 예시로, 과부하 증폭기(overloaded amplifier)를 통해 pure tone이 재생될 때 등이라고 함.

**Quantum mechanics**
- 양자역학에서 관측가능한 이산 스펙트럼은 관측 가능을 모델링하는데 사용되는 operator의 eigenvalue에 해당된다고 함.
  - Functional Analysis에 따르면, eigenvalue들은 isolated points들의 이산 집합이라고 함.



---
**Continuous vs Discrete Spectrum?**

위를 읽고 내가 이해한 바는, 데이터를 수집할 때 compact하냐 아니냐의 차이같음.
그래서 continuous 문서에선 general한 얘기, discrete 내용이 껴있고
discrete 문서에선 자세한 예시 및 기원에 대한 얘기가 상당 부분 존재

그리고, 전반적으로 `eigenvalue`를 중심적으로 언급하고 있음
왜 그럴까? 수학적으로 이해하자.

---

### Electromagnetic spectrum
전자기 스펙트럼. 감마선, X-ray, 저주파 고주파 등의 전자기선에 대한 내용들.

electromagnetic radiation, wavelengths, photon energies의 주파수 범위라고 함.

Maxwell's equation 내용도 나옴.

### Spectral Density
시계열 spectra와 신호 처리에 관련된 내용.

spectral density는 frequency에 대한 함수이고 time에 대한 함수가 아님.
그러나 긴 signal의 small window 크기의 spectral density를 계산하고 window에 관련된 시간에 대한 그림으로 그릴 수 있다. 이 graph를 `spectrogram`이라 부름. 이는 `short-time Fourier transform`과 `wavelets`과 같은 spectral analysis techniques의 기반이 됨.

`"Spectrum"`이 일반적으로 의미하는 바는 주파수에 대한 신호 분포를 묘사하는 power spectral density라고 함.

## Mathematics

### Pseudospectrum
pseudo spectrum을 아는 것은 non-normal operators과 eigenfunction을 이해하는 tool이 된다네.

행렬 $A$의 $\varepsilon$-pseudospectrum은 $A$와 $\varepsilon$-close한 행렬들의 모든 eigenvalues를 포함한다. 즉,
$$\Lambda_\varepsilon(A)=\{\lambda\in\mathbb{C}|\exist x\in\mathbb{C}^n\setminus\{0\},\exist E\in\mathbb{C}^{n\times n}:(A+E)x=\lambda x,\Vert E \Vert \leq \epsilon\}$$

[행렬의 eigenvalue를 계산하는 Numerical algorithms들](https://en.wikipedia.org/wiki/Eigenvalue_algorithm)은 rounding 및 기타 오차 때문에 근사적인 결과만을 제공하고 이러한 에러를 위의 행렬 $E$로 묘사한다고 함.

---
**[non-normal operator](https://en.wikipedia.org/wiki/Normal_operator)**
functional analysis에서 complex Hilbert space $\mathcal{H}$ 위에서 정의된 normal operator는 continuous linear operator $N:\mathcal{H}\rightarrow\mathcal{H}$임.
- 위 continuous linear operator는 hermitian adjoint $N^\ast$를 계산함
- 즉, $NN^\ast=N^\ast N$

Normal operator는 spectral theorem에서 이들을 다루기에 중요함.
normal operator의 예시는 아래와 같음.
- Unitary operators: $N^\ast =N^{-1}$
- Hermitian operators(i.e., self-adjoint operators): $N^\ast = N$
- Skew-Hermitian operators: $N^\ast = -N$
- positive operators: $N=MM^\ast\;\;for\;some\;M$ (so, $N$ is self-adjoint)

normal matrix는 Hilbert space $\mathbb{C}^n$위의 normal operator의 행렬 표현.

---
**[Spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem)**

Linear algebra와 Functional analysis에서 spectral theorem은 linear operator 혹은 행렬이 대각화 가능할 때에 대한 결과.
- 즉, 이를 몇몇 basis로 diagonal matrix로 표현할 수 있다는 얘기.

대각화 가능한 행렬을 포함하는 계산은 종종 대각행렬을 포함한 연산같이 연산을 간단하게 줄여주므로, 위 정리는 매우 유용하게 사용될 수 있다고 한다.

여러 공간에서 효과적으로 사용, 핵심은 대각화!

spectral theorem은 canonical decomposition, spectral decomposition, eigenvalue decomposition, eigendecomposition이라 불리는 분해법을 제공한다고도 하네.

더러운 코시가 여기서도 등장. 모든 real symmetric matrix는 대각화 가능하다. 코시는 determinants를 처음으로 시스템화시키기도 함.
John von Neumann이 spectral theorem을 일반화.

>Finite-dimensional case

에르미트 행렬(Hermitian matrix), 대칭 행렬(Symmetric matrix)에 대해 고려.
$A=A^\ast\;\equiv\;\;<Ax,y>=<x,Ay>$
위 Hermitian condition이 의미하는 것은 Hermitian map의 모든 eigenvalue는 실수값이라는 것. $Ax=\lambda x$

$Theorem.\text{ Spectral Theorem}$
$\quad\quad\text{If }A\text{ is Hermitian, }\exist\;orthonormal\;basis\text{ of }V\text{ consisting of }eigenvectors\text{ of }A.$
- $\quad\text{Each }eigenvalue\text{ is real.}$
- $\quad V\text{ is finite-dimensional complex inner product space}$
  $\quad\text{endowed with a positive definite sesquiliear inner product }<\cdot,\cdot>$

**spectral decomposition**

Let $V_\lambda = \{v\in V\; : \; Av=\lambda v\}$ be `eigenspace` corresponding to an `eigenvalue` $\lambda$.
$V$는 $V_\lambda$의 orthogonal direct sum. 즉,
$$V=V_{\lambda_1}\perp V_{\lambda_2} \perp \dots \perp V_{\lambda_m}$$

다른 말로, $P_\lambda$를 $V_\lambda$의 orthogonal projection이라 하고 $\lambda_1,\dots,\lambda_m$을 $A$의 eigenvalue라 하자.
이 때 spectral decomposition을 아래와 같이 쓸 수 있다.
$$A=\lambda_1 P_{\lambda_1}+\cdots+\lambda_m P_{\lambda_m}$$

이는 polynomial function $f$를 가미하여 아래와 같이 쓸 수도 있음
$$f(A)=f(\lambda_1) P_{\lambda_1}+\cdots+f(\lambda_m) P_{\lambda_m}$$

spectral decomposition은 Schur decomposition과 singular value decomposition의 특수한 경우라고 함.

> Normal matrices

Let $A$ be an operator on a finite-dimensional inner product space.

$$A\text{ is said to be }normal\quad\text{if }A^\ast A=AA^\ast$$

$$A\text{ is }normal\;\iff\; A\text{ is unitarily diagonalizable}$$

By the `Schur decomposition`, $A=UTU^\ast$ where $U$: unitary and $T$: upper-triangular.
If $A$ is normal, then $T^\ast T=TT^\ast$.
```
왜냐고?
A = UTU*
If A is normal, then A*A=AA*
=> (UTU*)*(UTU*) = (UTU*)(UTU*)*
=> UT*U*UTU* = UTU*UT*U* (since (U*)*=U)
=> UT*TU* = UTT*U*       (since U is unitary)
=> T*T = TT*             (since unitary is invertible)
```
Hence, $T$는 diagonal. `(since normal upper triangular matrix is diagonal)`
역 방향은 자명함. ($A=UTU^\ast$일 때, $T^\ast T=TT^\ast$이면 $A$는 normal.)

즉, 아래와 같은 말.
$$A\text{ is }normal\;\iff\;\exist\;unitary\;matrix\;U\;s.t\;A=UDU^\ast$$
- where $D$ is a diagonal matrix.
- D의 대각성분은 $A$의 eigenvalue들.
- $U$의 열벡터들은 $A$의 eigenvector들이고 orthonormal함.
- Hermitian과 다르게 $D$의 대각성분이 실숫값일 필욘없음.

(그 외 여러 경우가 있지만, 핵심 컨셉은 대각화!)

---

### Spectrum of a matrix

### Spectrum (functional analysis)

### Spectrum (topology)

### Spectrum of a $\text{C}^\ast$-algebra와

### Spectrum of a graph

### Spectrum of a ring

### Spectrum of a sentence

### Spectrum of a theory

### Stone space

결국 핵심 개념은 대각화!

향후 수학 공부 및 spectrogram 등을 공부하여 내용 추가 정리하겠음.


#### 출처
- https://en.wikipedia.org/wiki/Spectrum
- https://en.wikipedia.org/wiki/Spectrum_(disambiguation)
