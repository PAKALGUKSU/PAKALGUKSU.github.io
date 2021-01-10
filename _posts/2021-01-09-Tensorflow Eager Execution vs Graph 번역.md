---
title: "Tensorflow Eager Execution vs Graph (@tf.function) 번역"
categories:
    - Deep Learning
    - Tensorflow
toc: True
---

이 글은 허락 하에 Jonathan Hui의 글 [TensorFlow Eager Execution v.s. Graph (@tf.function)](https://jonathan-hui.medium.com/tensorflow-eager-execution-v-s-graph-tf-function-6edaa870b1f1)을 번역 및 내용을 추가한 글입니다. 내용이나 번역에 오류가 있으면 댓글로 남겨주세요.

## Eager Execution
TF2에서는 eager execution의 사용을 적극 권장하고 있다. Eager execution은 코딩과 디버깅을 쉽게 해주기 때문이다. 그러나 실제 training이나 production에서도 항상 그것의 사용이 권장되지는 않는다. 이 글에서는 tensorflow의 두 모드 : eager execution mode, graph mode와 각각의 장단점에 대해 소개한다. 특히, graph mode에서 코드를 실행하는 것은 생각과 같이 잘 동작하지 않을 때가 있다. 만약 중요한 몇 가지 사항들을 고려하지 않는다면 (graph mode에서) 심각한 성능 문제가 발생할 수 있다.

기본적으로, TF 2.x 버전의 operation들은 **eager execution** 모드로 실행된다. 예를 들어, 아래의 `tf.matmul`(행렬곱 연산)은 즉시 실행되어 값으로 `[[4.]]` 가지는 `tf.Tensor` object를 return한다. 이것은 우리가 보통 생각하는 python 코드의 동작과 유사하다. 코드를 줄 단위로 실행되며 연산 결과가 즉시 return 된다.

```python
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
```
하지만, graph mode에서의 동작은 다르다. 위와 같이 동작하는 대신, `tf.matmul`은 computational graph의 node를 가리키는 symbolic handle(node에 접근할 수 있게 해주는 변수)을 return한다. 즉, 행렬곱 연산은 즉시 수행되지 않는다.

## Eager Execution의 단점
Graph mode에서 `tf.matmul`은 computational graph(`tf.Graph`)에 computational node(s)(`tf.Operation`)을 추가한다. TF v1에서는 `session.run`을 호출해 computation graph을 compile하고 실행할 수 있었다. 이러한 지연된 실행은 TF Grappler가 백그라운드에서 자동으로 동작할 수 있게 한다. TF Grappler는 성능 향상을 위해 [다양한](https://www.tensorflow.org/guide/graph_optimization) graph 최적화를 수행한다. 예를 들어, node operation은 효율성을 위해 합쳐지거나 제거될 수 있다. TF 2.x에서 이런 장점을 이용하기 위해서는 eager execution이 아닌 graph mode에서 코드를 실행해야 한다. TF의 내부 벤치마크에 따르면 graph 모드에서 평균 15%의 성능 향상이 이루어진다. ResNet50과 같이 무거운 연산이 많은 모델에서는 eager execution(GPU)와 graph mode의 속도가 비슷하지만, 가벼운 operation이 굉장히 많은 모델의 경우는 차이가 커진다. CNN과 같이 적은 숫자의 무거운 연산이 많은 경우는 차이가 줄어든다. 즉, 모델에 따라 성능의 차이가 존재한다.

코드를 graph mode로 실행하는 방법은 간단하다. 추가적인 코드 수정 없이 함수에 `@tf.function`이라는 annotation을 추가하면 함수 전체가 compile 및 최적화된 이후 하나의 computational graph로 실행된다. `@tf.function`은 함수 내에서 호출하는 모든 함수에 대해 적용되며 하나의 graph를 생성한다. 

```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training = True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

Graph mode는 python 코드에서 dataflow를 생성한다. 생성된 그래프는 이식 가능하며, 모델의 python 코드가 없는 환경에서 모델을 불러오거나 python 환경이 없는 기기에서 배포할 수도 있다. 이 조건은 `SavedModel` 파일로 모델을 저장하기 위해 필수적이다. 이런 이식성은 배포시 큰 장점을 갖는다. 데이터 전처리를 포함한 `SavedModel`로 모델을 저장함으로써 배포 환경에서 데이터 전처리 코드를 작성할 때 실수 발생을 없앨 수 있다. 이러한 데이터 전처리 과정은 training data에 매우 민감하다. 예를 들어, `TextVectorization` 레이어는 반드시 train dataset에 있는 단어들로 초기화되어야 하는데, 이는 배포 환경에서 실수할 여지가 많은 부분이다.

## Graph Mode의 단점
하지만, Graph Mode에도 심각한 단점이 있다. TF 2에서 eager execution 모드를 기본으로 사용하는 이유는 코딩과 디버깅이 쉽기 때문이다. TF 1 코드들은 지루하고 디버깅하기 힘들다. Graph Mode에서 `tf.matmul`은 연산 결과를 즉시 return하지 않고 computational graph에 node를 추가한다. 따라서, graph mode의 경우 디버거가 `tf.matmul`에 breakpoint를 걸어 멈추게 할 수 없고, 코드를 추적하기 힘들다.

이러한 이유로 초기 개발과 디버깅 과정에서는 annotation을 주석 처리할 수 있다. 또는, `tf.config.run_functions_eagerly(True)`를 사용해 eager execution 모드를 실행할 수 있다. 아래 코드의 square 함수 전에 True를 설정하고 함수 이후 False를 설정하면 `@tf.function` 처리된 함수 안에서 디버깅을 할 수 있다.

![Debuggin in @tf.function](/images/tf_eager_vs_graph/debugging_in_tfgraph.png){: width="80%" height="70%"}{: .align-center}

Eager execution을 사용하면 일반적인 python 문법을 통해 코딩할 수 있기 때문에 코드가 python에 친화적이고 읽기 편하다. 하지만, 같은 코드를 graph mode에서 사용하기 위해서는 코드들을 graph에 포함될 수 있게 수정해야 한다. AutoGraph(뒤에서 다시 다룸)은 몇몇 python 제어문(if, while, for)들을 TF operation으로 자동 변환해준다. 하지만, 이러한 자동 변환의 경우 이해하기 어려운 불규칙적인 변환이 일어날 수도 있기 때문에 특정 코드가 무시되거나 오류가 발생할 수도 있다. 또는, 예상하지 못한 영향이 발생할 수도 있다. 앞으로의 글은 graph mode에서 발생할 수 있는 문제들에 대해 다룬다.

## Issues in Graph Mode
Graph mode에서 발생할 수 있는 이슈들.

### Assert 사용하지 않기
몇몇 python 문법들은 graph mode에서 지원되지 않는다. 예를 들어, `@tf.function` 함수 안에 있는 `assert` 구문은 exception을 발생시킨다. 따라서, 두 모드 모두에서 `tf.debugging.assert_{condition}`을 사용한다.

```python
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)

        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (32, 10))
```

### Tracing 이란
그렇다면, graph mode에서 python 및 TF 코드는 어떻게 graph로 변환되는 것일까? `@tf.funtion`으로 annotate된 함수가 처음 실행되면, 해당 함수는 computation graph로 변환되기 위해 먼저 **trace**된다. 개념적으로, 함수는 graph로 compile 되는 것이다. Graph가 생성되고 나서는 자동적으로 실행된다. 완벽한 설명은 아니지만, 간단한 예제를 통해 자세히 살펴보자.

### print v.s. tf.print
Python의 `print`는 parameter를 콘솔에 출력한다. 하지만 `@tf.function` 함수 안에서 `print`는 tracing(graph compile)과정에서만 실행되고, graph에 아무런 node도 추가하지 않는다. 따라서, 이러한 동작은 tracing 과정에서는 동작하지만 graph 실행 과정에는 영향을 미치지 못하는 **Python side effect**로 불린다. 이와 달리, `tf.print`는 TF operation이며 tracing 과정에서 콘솔에 아무것도 출력하지 않고 graph에 node를 추가한다. 해당 함수는 graph가 실행될 때 결과를 콘솔에 출력한다. 이러한 이유로, trace와 실행 과정에서의 차이를 파악하고 문제를 해결하기 위해 해당 연산들을 사용할 수 있다.

함수 `f`안의 operation들은 graph를 compile하기 위한 코드임에도 불구하고 tracing 과정에서 실행된다. 함수 안의 operation들은 python operation과 TF operation으로 나눌 수 있다. 물론 모든 operation들은 python으로 실해오디지만, TF operation은 실제 연산을 수행하지 않고 단지 graph에 node를 추가할 뿐이다. `@tf.function` annotation 덕분에 tracing이 끝나면 graph는 자동적으로 수행된다. 예시를 통해 살펴보자.

![Trace and execute](/images/tf_eager_vs_graph/trace_execution.png){: width="80%" height="70%"}{: .align-center}

(출력 결과가 생각한 순서와 조금 다를 수 있기 때문에, 디버거를 통해 코드 순서에 따라 실행한 결과이다)

Line 23에서 `f(1)`을 처음 실행했을 때, 먼저 graph를 만들기 위해 함수가 trace 된다. Tracing 과정에서 `print`가 ①을 출력하고 `tf.print`는 graph에 node를 추가하는 것 외에 다른 동작을 수행하지 않는다. Graph가 생성되고 나면, 실행된다. `print`는 graph 안에 존재하지 않으므로 아무것도 출력하지 않고, `tf.print`는 ➁를 출력한다. 즉, line 23은 서로 다른 단계에서 콘솔에 두 줄을 출력하게 된다.

Line 25에서 `f(1)`을 다시 호출할 때 함수는 이미 trace 되어있으므로 graph는 바로 사용될 수 있다. 따라서 즉시 graph가 실행되며 ③이 출력된다.

### Autograph (tf.autograph)
Eager execution은 `while`, `for`, `if`, `break`, `continue`같은 python 제어문을 사용할 수 있게 해준다. 해당 코드들을 graph mode에서도 호환되게 하기 위해 AutoGraph는 몇몇 제어문을 TF operation으로 자동 변환 해준다. 변환된 코드는 python operation이 아닌 TF operation으로 취급된다. 이 기능은 python에 가까운 코드들을 두 모드 모두에서 사용할 수 있게 해준다. 아래는 AutoGraph를 이용해 python 코드를 TF operation으로 변환하는 예시들이다.

![AutoGraph](/images/tf_eager_vs_graph/autograph.png){: width="80%" height="70%"}{: .align-center}

AutoGraph는 for loop으로 작성된 dataset iteration도 TF operation으로 변환해준다.

```python
@tf.function
def train(model, dataset, optimizer):
    for x, y in dataset:
        with tf.GradientTape() as tape:
            # training = True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            prediction = model(x, training=True)
            loss = loss_fn(prediction, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

그러나 아직 고려해야 할 아주 중요한 이슈들이 있다. 해당 내용은 tracing에 대해 더 자세히 소개한 이후로 미루겠다.

### 'if' 문의 tracing
`while`이나 `if`의 조건에 해당하는 변수가 Tensor로 주어지면 아래와 같은 꽤 놀라운 변환이 일어난다. *n*은 Tensor이기 때문에 `if n==0`이라는 코드는 해당하는 TF operation인 `tf.conf`로 변환된다.

![if to cond](/images/tf_eager_vs_graph/if_to_cond.png){: width="80%" height="70%"}{: .align-center}

그런데 왜 trace 과정에서 세 줄(①, ➁, ③)이나 출력될까?

Computational graph가 다양한 값을 가진 입력 Tensor에서도 동작할 수 있게 하기 위해, TF는 사실 모든 분기를 trace 한다. 따라서, 위의 세 분기 모두가 호출되고 각 분기가 각각 한 줄씩을 출력하는 것이다. 이러한 동작은 trace 결과를 재활용하기 좋게 한다. 같은 shape을 가진 Tensor를 입력으로 `f`를 두 번째로 호출할 때, 추가적인 tracing이 필요하지 않기 때문이다. 생성된 그래프는 이미 다양한 값`n`에 맞게 동작한다.

만약 `f(1)`처럼 입력 `n`이 스칼라 값이라면 어떻게 될까? 입력이 스칼라인 경우에 AutoGraph는 `if`를 `tf.cond`로 변환하지 않는다. Tracing 과정에서 `if`문은 원래대로 동작한다. `n`이 1인 경우, 분기 `elif n==1`은 trace 되지만 다른 분기들은 tracing되지 않으며 `if`문은 graph에 추가되지 않는다.

![Trace Scalar](/images/tf_eager_vs_graph/trace_scalar.png){: width="80%" height="70%"}{: .align-center}

위 함수는 아래와 같이 tracing 된다.

![Trace Scalar same as](/images/tf_eager_vs_graph/trace_scalar_same.png){: width="80%" height="70%"}{: .align-center}

즉, `if ... elif`문은 python side effect이다. 

그렇다면 다른 분기의 코드가 필요한 `f(2)`가 호출 되었을때는 어떻게 될까? 다행히 코드는 다시 tracing되어 `f(2)`를 위한 새로운 graph를 생성한다. 자세한 설명과 그 영향은 이후에 다룬다.

### 'while', 'for'문의 tracing
`while`문을 통해 다시 내용을 확인해보자. 조건에 Tensor가 사용된다면 코드는 `tf.while_loop`로 변환되며 내용은 한번 trace 된다. 만약 Tensor가 아니라면, python의 `while`문 처럼 동작한다. 아래 예시에서 볼 수 있듯이, 코드는 3번 반복되며 각 반복마다 graph를 업데이트한다.

![Tracing while](/images/tf_eager_vs_graph/trace_while.png){: width="80%" height="70%"}{: .align-center}

만약 `f(4)`로 다시 호출된다면, `f`는 아래와 같이 다시 trace 된다.

![f(4) trace](/images/tf_eager_vs_graph/f4_trace_while.png){: width="80%" height="70%"}{: .align-center}

`for`문에서도 비슷한 변환이 일어난다. `for i in expression`이 Tensor로 호출된다면 해당 코드는 `tf.while_loop`로 변환된다. `tf.range`는 Tensor를 return하므로, 아래의 for loop은 대체된다.

```python
for i in tf.range(n):
```

아래 코드는 scalar에 대해 실행될 때 trace가 어떻게 달라지는지 보여준다.
![Tracing for](/images/tf_eager_vs_graph/f4_trace_while.png){: width="80%" height="70%"}{: .align-center}

### Dataset v.s. NumPy ndarray
학습 과정이 아래와 같이 `tf.function`화 되었다면, `for`문의 `in`에 해당하는 부분이 python이나 NumPy 변수가 아닌 dataset(`tf.data.Dataset`)이어야만 하는 경우가 있다. Python 혹은 NumPy 변수의 경우, tracing 과정에서 모든 iteration이 graph에 node를 추가할 수 있다. 따라서, 아주 많은(반복하는 데이터셋 개수 만큼) node들이 추가될 수 있다. 하지만 `dataset`의 경우는 `tf.data.Dataset` 관련된 operation이 graph에 한 번만 추가된다(모든 iteration마다 추가되지 않음). TensorFlow loop는 loop 내용을 한 번 trace하며 실행 과정에서 반복할 iteration 횟수를 동적으로 결정한다.

```python
@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
    return loss
```

### Python List v.s. TensorArray
Python list는 graph mode에서 잘 지원되지 않는다. `@tf.function` 함수 안 혹은 밖에서 list가 수정되는 경우는 더욱 그렇다. 필자는 annotate된 함수 안에서 python list를 사용하지 말아야 할 너무 많은 경우들을 확인했다. 

```python
l = []

@tf.function
def f(x):
    for i in x:
        l.append(i+1) # Caution! Will onyl happen once when tracing

f(tf.constant([1, 2, 3]))
print(l)
```

예를 들어, `l.append` 연산은 python runtime에 의해 수행되기 때문에 graph에 node를 전혀 추가하지 않는다. 이는 python 문법이 graph execution에서 무시되며 tracing 과정에서 예상하지 못한 동작을 하는 예시 중 하나이다. 만약 runtime에 데이터를 추가하는 list와 유사한 자료구조가 필요하다면, TensorArray를 사용해야 한다. 이는 RNN에서 매 timestep마다 hidden state를 추가해야하는 경우에 꽤 흔한 상황이다.

```python
@tf.function
def f(x):
    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    for i in range(len(x)):
        ta = ta.write(i, x[i] + 1)
    return ta.stack()

f(tf.constant([1, 2, 3]))
```

### Polymorphic과 코드 성능에 관해
Python은 type에 민감한 언어가 아니다. 즉, 함수를 실행할 때 마다 서로 다른 type을 가진 parameter를 전달 할 수 있다. 그러한 상황에서 parameter를 다루는 것은 호출된 함수의 역할이다. 반면, TensorFlow는 상당히 정적(static)이다. Graph를 만들기 위해서는 parameter 들의 datatype과 shape 정보가 필요하다. 실제로 TF는 보다 효율적인 실행을 위해 서로 다른 datatype이나 shape를 가진 parameter를 받는 경우 서로 다른 graph를 생성한다. 입력 Tensor의 shape만이 변하더라도, tracing이 다시 수행될 수도 있다.

`f.get_concrete_function`은 `ConcreteFunction`을 return하는 함수로, computational graph를 나타내는 `tf.Graph`에 대한 warpper 함수 이다. 아래 예시에서, `f1`과 `f2`는 다른 shape를 가진 input tensor를 입력으로 받는다. 따라서, 두 함수에 대한 `ConcreteFunction`은 서로 다른 두 graph를 가진 것과 같이 고려되어 서로 다르다. 다행히, 이들은 `Function`(`python.eager.def_function.Function`)으로 wrapping 되어 있기 때문에 내부적인 차이에 대해 고민할 필요 없이 `Function` 함수를 통해서 호출할 수 있다.

```python
@tf.function
def f(a, b):
    return a + b

f1 = f.get_concrete_function(tf.constant([1, 2]), tf.constant([3, 4]))
f2 = f.get_concrete+function(tf.constant([1, 2, 3]), tf.constant([3, 4, 5]))

print(f1 is f2) # False

print(f(tf.constant([1, 2]), tf.constant([3, 4])))
print(f(tf.constant([1, 2, 3]), tf.constant([4, 5, 6])))
```

만약 이 둘이 서로 같은 graph를 사용하도록 강제하고 싶다면, general한 shape를 가진 `TensorSpec`를 포함하는 `input_signature`를 추가할 수 있다. 예를 들어, shape를 `None`으로 명시함으로써 아래 벡터와 행렬에서 동일한 graph를 사용할 수 있다. 하지만 graph가 덜 효율적일 가능성이 존재한다.

```python
@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(x):
    return x + 1

vector = tf.constant([1.0, 1.0])
matrix = tf.constant([[3.0]])
print(f.get_concrete_function(vector) is f.get_concrete_function(matrix)) # True
```

아래의 `None` dimension은 일종의 와일드카드로, 함수들이 동적 길이의 input들에 대해 tracing된 결과를 재활용 할 수 있게 해준다.

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def g(x):
    print('Tracing with', x)
    return x

# No retrace!
print(g(tf.constant([1, 2, 3])))
print(g(tf.constant([1, 2, 3, 4, 5])))
```

본 적 없는 data type이나 shape의 parameter때문에 함수가 retrace 되는 경우, 추가적인 오버헤드가 발생한다. 특히, input parameter가 스칼라인 경우 문제가 생길 수 있다. TF는 서로 다른 스칼라 값이 입력될 때 마다 retrace를 수행한다. 아래와 같이, `f3`은 서로 다른 스칼라 input을 가지기 때문에 함수는 retrace되고, `f1`, `f2`와 다른 graph를 가지게 된다. 이러한 작동은 모순적으로 TF가 앞에서 소개한 `if`, `while` 조건에 스칼라 값이 입력되는 경우를 다룰 수 있게 해준다. 

```python
@tf.function
def f(a, b, step):
    tf.print(step)
    return a + b

f1 = f.get_concrete_function(tf.constant([1, 2]), tf.constant([3, 4]), 1)
f2 = f.get_concrete_function(tf.constant([1, 2]), tf.constant([3, 5]), 1) # Same
f3 = f.get_concrete_function(tf.constant([1, 2]), tf.constant([3, 5]), 2) # Different

print(f1 is f2) # True
print(f1 is f3) # False
```

오버헤드를 줄이기 위해서는 함수를 잘 설계해야 한다. 예를 들어, 개발자는 단순하게 training step 값을 스칼라로 넘겨줄 수 있다. 이러한 코드는 수많은 retracing을 유발하며, 성능 저하가 생긴다. 이러한 문제를 피하기 위해, 스칼라 대신 Tensor를 사용해야 한다.

```python
def train_one_step():
    pass

@tf.function
def train(num_steps):
    print("tracing with num_steps = ", num_steps)
    tf.print("Executing with num_steps = ", num_steps)
    for _ in tf.range(num_steps):
        train_one_step()

print("Retracing occurs for different Python arguments.")
train(num_steps=10)
train(num_steps=20)

print()
print("Traces are reused for Tensor arguments.")
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```

### Graph는 trace 순간의 Snapshot이다
Graph를 생성하기 위한 trace가 일어날 때, 그 순간 함수의 snapshot을 만들게 된다. 따라서, 아래 코드에서 `f`를 다시 호출하기 전 list `l`이 변했지만, 함수는 기존 `l`을 참조하게 된다. (하지만 위에서 소개한 대로 list를 사용하는 것은 피해야 한다)

```python
@tf.function
def f(x):
    for i in l:
        tf.print(i)

f(tf.constant([1, 2]))
l = l.append(3)
f(tf.constant([1, 2]))
```

### Iterator
Generator와 Iterator과 같은 다양한 python 기능들은 상태에 대한 정보를 추적하기 위해 python runtime에 의존한다. 따라서, graph mode에서 실행하는 경우 graph는 이러한 변화를 알지 못한다. 아래에서 볼 수 있듯이, iterator는 여러번 호출되어도 제대로 반복을 수행하지 않는다. Graph는 trace 되는 순간의 iterator 값의 snapshot을 가지고 있을 뿐이다.

```python
external_var = tf.Variable(0)

@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var:", external_var)

iterator = iter([0, 1, 2, 3])

buggy_consume_next(iterator) # 0
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator) # 0
buggy_consume_next(iterator) # 0
```

### Variable은 처음 실행에 한번만 생성할 수 있다
`tf.Variable` 변수는 함수가 처음 실행될 때만 생성할 수 있다. 아래 코드에서 line 15가 없다면 line 16에서 함수의 첫 실행이 아닐 때 변수를 생성하게 되며, exception이 발생할 것이다. 이러한 연산은 graph가 생성된 이후, graph를 수정하려는 시도이다. TF graph는 static하기 때문에, 이와 같은 수정을 허용하지 않는다. 이런 방법 대신 (적용 가능한 상황이라면) 함수 바깥에서 모델과 관련 없는 변수를 선언하고 parameter로 넘겨주는 방법을 사용할 수 있다.

```python
class MyModule(tf.Module):
    def __init__(self):
        self.v = None
    
    @tf.function
    def __call__(self, x):
        if self.v is None:
            self.v = tf.Variable(tf.ones_like(x))
        return self.v * x
```

### Model Training
`model.compile`에서 설정을 통해 eager execution을 비활성화 할 수 있다. (비활성화 한다면) `model.fit`가 실행될 때, 모델은 trace 되어 graph mode로 실행된다.

```python
model.compile(run_eagerly=False, loss=loss_fn)

model.fit(input_data, labels, epochs=3)
```

### Tensor objects 와 NumPy 사이의 변환
Eager execution을 사용한다면, NumPy 연산은 `tf.Tensor`를 입력으로 받을 수 있다

```python
c = np.multiply(a, b)
```
반대로, `tf.math` 연산은 python object나 NumPy array를 `tf.Tensor` object로 변환한다. `tf.Tensor`를 NumPy `ndarray`로 명시적으로 변환하고 싶다면, `numpy()`를 사용한다.

```python
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

### 함수를 graph function으로
Python 함수는 annotation 없이 graph 와 같이 실행될 수 있다. 아래 `tf_function`은 함수를 `python.eager.def_function.Function`으로 변환하며, 이는 `@tf.function` annotation에서 소개된 함수와 동일한 class이다.

```python
# Define a Python function
def function_to_get_faster(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x

# Create a 'Function' object that contains a graph
a_function_that_uses_a_graph = tf.function(function_to_get_faster)

# Make some tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

# It just works!
a_function_that_uses_a_graph(x1, y1, b1).numpy()
```

### Graph에서 python 코드 실행하기
Graph mode에서는 일반적으로 graph 실행을 위해 모든 operation을 python에 독립적인 graph 형태로 변환하려고 한다. 그러나, 만약 graph 안에서 python 코드를 실행하고 싶다면 `tf.py_function`을 사용할 수 있다. `tf.py_function`는 모든 입력, 출력 변수를 tensor로 변환한다. 하지만, graph의 이식성을 잃게 되며 여러 개의 GPU를 사용하는 distributed 환경에서 잘 동작하지 않을 수 있다. 아래는 python list를 다루는 코드이며, `py_function`을 이용했다.

```python
external_list = []

def side_effect(x):
    print('Python side effect')
    external_list.append(x)

@tf.function
def f(x):
    tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)
# The list append happens all three times!
assert len(external_list ) == 3
# The list contains tf.constant(1), not 1, because py_function casts everything to tensors.
assert external_list[0].numpy() == 1
```

이러한 코드의 사용은 최대한 피해야 하지만, 이미지 augmentation 과정에서 `scipy.ndimage`와 같은 외부 라이브러리를 사용하는 경우 종종 사용되곤 한다.

```python
def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image
```

예제에서는 이미지 augmentation을 위해 `scipy.ndimage`의 arbitrary rotation을 사용한다.

```python
def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label

rot_ds = images_ds.map(tf_random_rotate_image)

for image, label in rot_ds.take(2):
    show(image, label)
    plt.show()
```

### Retracing을 줄이기 위한 type annotation
스칼라 input의 경우, retracing을 줄이기 위한 experimental 기능이 있다. 예를 들어, `tf.Tensor` 로 annotate된 input의 경우 non-Tensor 값이더라도 Tensor로 변환된다. 따라서, `f(scalar)`는 서로 다른 값들을 입력받더라도 retrace 되지 않는다. 아래 코드에서 `f_with_hints(2)`는 retracing을 일으키지 않는다.

![Annotation for no retracing](/images/tf_eager_vs_graph/type_anno_no_retrace.png){: width="80%" height="70%"}{: .align-center}


## Thoughts
이 글에 다룬 이슈들은 반드시 따라야만 하는 디자인 규칙보다 현재 TF에서의 구현상 한계(다른 스칼라 값 input의 경우 retracing이 수행되는)에 가깝다. 때때로 코드의 동작을 이해하거나 설명하기 힘든 경우들도 있다. TF는 지속적으로 수정되고 있으니 코드를 작성할 때(특히 이상하다는 생각이 들면) TF의 최신 document를 찾아보는 것을 추천한다. 다행히 최근 모델을 코딩할 때는 이 글에서 다룬 까다로운 경우들을 고려하지 않아도 되는 경우가 많다.

## Credits & References
[Introduction to graphs and tf.functions](https://www.tensorflow.org/guide/intro_to_graphs)

[Better performance with tf.function](https://www.tensorflow.org/guide/function)

[tf.function](https://www.tensorflow.org/api_docs/python/tf/function)

[Jonathan Hui의 원문 글](https://jonathan-hui.medium.com/tensorflow-eager-execution-v-s-graph-tf-function-6edaa870b1f1)