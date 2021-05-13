---
title: "API의 뜻에 대한 간단한 비유 : API는 은행 창구이다"
categories:
    - Development
tags :
    - Development
    - Dev-Basic
toc: True
---
## API란?
[위키피디아의 API 문서](https://ko.wikipedia.org/wiki/API)에 따르면 API(Application Programming Interface)의 정의는 *응용 프로그램에서 사용할 수 있도록, 운영 체제나 프로그래밍 언어가 제공하는 기능을 제어할 수 있게 만든 인터페이스*를 의미한다... 지만 API가 무엇인지 모르는 사람은 보통 이 한 줄을 읽고 API의 뜻을 이해할 수 없을 것이다. API에 대한 간단한 비유는 **은행 창구**이다.

## 은행과 은행 창구
은행에 가서 돈을 인출하기 위해서는 창구에서 직원에게 원하는 업무를 말한 뒤 직원이 요구하는 정해진 절차(신분증 제시, 문서 작성 등)를 밟고 기다려야 한다. 여기서 **창구 및 정해진 절차**가 API라고 볼 수 있다.

돈을 인출하기를 원하는 사람이 직접 은행 금고에 들어가서 자기 돈을 찾아간다면 어떻게 될까? 그 은행에 돈을 맡기지 않은 사람도 마음대로 돈을 훔쳐갈 수도 있고, 대기열 관리도 잘 되지 않아서 간단한 업무를 수행할때도 시간이 훨씬 오래 걸릴 수도 있고, 은행에서도 각 고객이 입출금 내역을 관리하기 힘들 것이다.

따라서, 은행은 금고 등 은행에서 관리하는 내부 자산에 접근할때 반드시 창구 등을 통해 정해진 절차를 밟고 고객이 직접 일을 처리하는 것이 아니라 은행 직원이 해당 업무를 대신 수행해주는 방식으로 업무를 수행한다.

## 컴퓨터의 '은행 창구'
프로그래밍에서도 마찬가지이다. OS에서 제공하는 kernel API는 system resources(CPU, RAM 등)에 사용자가 마음대로 접근해 자원을 사용하는 것을 방지하고, 효율적인 process scheduling을 수행한다. Web에서 제공하는 API 역시 사용자가 서버 내부 데이터베이스등에 직접 접근하는 대신, 정해진 URL과 형식에 맞춰 사용자가 요청을 보내면 필요한 작업을 수행한 이후 사전에 약속된 결과를 반환해준다.

## Web API
API라는 단어가 가장 많이 쓰이는 분야 중 하나는 web이기 때문에, web의 경우에 대해 부연 설명을 추가한다. 내 블로그에 접속하기 위해서는, [블로그 링크](https://pakalguksu.github.io/)에 해당하는 주소의 웹 코드(HTML, CSS, JavaScript)를 해당 코드를 가지고 있는 서버에서 받아와야 한다. 그러한 요청을 보내기 위해서 Chrome등의 웹 브라우저는 정해진 규칙에 따라 서버에 필요한 정보를 보내고 사이트 코드를 보내달라는 요청을 보내고, 서버는 정해진 형식에 맞춰 사이트 코드를 반환해준다. Chrome은 이 반환된 코드를 렌더링해 웹 브라우저에 보여주는 것이다.

아래 코드는 블로그 링크에 해당하는 글을 가져오기 위한 javascript fetch를 이용한 request 코드를 Chrome 개발자 도구를 이용해 확인해본 것이다. 
```javascript
fetch("https://pakalguksu.github.io/", {
  "headers": {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,zh;q=0.6,zh-CN;q=0.5,zh-TW;q=0.4,zh-HK;q=0.3",
    "cache-control": "max-age=0",
    "if-modified-since": "Thu, 18 Mar 2021 05:45:44 GMT",
    "if-none-match": "W/\"6052e908-31f7\"",
    "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"90\", \"Google Chrome\";v=\"90\"",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1"
  },
  "referrerPolicy": "strict-origin-when-cross-origin",
  "body": null,
  "method": "GET",
  "mode": "cors",
  "credentials": "include"
});
```

즉, backend를 코딩하는 경우는 위와 같이 정해진 URL/형식에 맞춘 요청이 들어오면 적당한 값을 반환할 수 있게 코딩하고 해당 API를 사용할 사람들에게 API 명세를 제공하고, frontend를 코딩하는 경우는 backend에서 제공하는 API 형식에 맞춰서 적절한 값들을 포함한 요청을 보내고 반환된 결과를 사용하면 된다.