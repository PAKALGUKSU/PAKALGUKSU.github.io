---
title: "SSH port forwarding을 이용해 원격 서버를 통해 다른 서버로 SSH 접속하기"
categories:
    - Development
tags :
    - Development
toc: True
---
## SSH port forwarding
SSH port forwarding (SSH Tunneling)은 특정 서버를 통해 다른 서버에 접속하고 싶을 때 사용할 수 있다. 최종적으로 내 컴퓨터에서 접속하고자 하는 서버가 보안 상 문제 등을 이유로 직접 ssh 접근을 허용하지 않고, 지정된 서버에서의 접속만 가능한 상황의 경우 유용하게 사용할 수 있다.

예를 들어, serverA (port 22)에 접속하고 싶은 상황에서 보안상 문제로 인해
```shell
ssh jongha@serverA
```
와 같은 접속은 불가능하고, 대신 serverB (port 22)에 접속해 해당 서버에서 serverA (port 22)에 접속하는 것만 가능한 경우를 생각해보자

```shell
ssh jongha@serverB
# logged in to server B
ssh jongha@serverA
```

물론 어떻게든 ssh 연결이 가능하기만 하면 되는 상황이라면 위와 같이 serverB에 접속한 이후 쉘에서 직접 ssh를 통해 다시 serverA에 접속해도 되지만, ssh를 두번 사용해야 한다는 불편함도 있고, PyCharm의 remote debugger등 쉘 환경에서 ssh를 두번 사용하는 방법으로는 접속이 불편한 경우에는 ssh 명령을 통해 직접 접속하는 방법이 필요하다.

ssh port forwarding 기능을 사용하면, 일종의 '터널'을 뚫어서
```
ssh jongha@serverA
```
를 입력했을 때 자동으로 serverB를 통해 serverA에 접속할 수 있도록 해준다.

## How to
구체적인 port forwarding 방법은 아래와 같다.

### ssh config
먼저, ssh config file을 찾는다 (Mac의 경우 home directory의 /.ssh/config).
파일 내용을 다음과 같이 수정하면 간단히 port forwarding 설정을 할 수 있다.
```
 Host serverB
    HostName serverB
    User jongha

 Host serverA
    HostName serverA
    ProxyCommand ssh jongha@serverB nc serverA 22
    User jongha
  ```