---
title: "note"
description: "copy了 claude-code from scratch 简单记一些东西吧"
pubDate: "Jun 27 2026"
category: "note"
mood: "-"
tags: ["agent"]
featured: false
---

通过这个项目，对于agent的工作流程更加熟悉了吧。

每一次调用api都是从零记忆开始的，所以为了实现交互，每一次对话都需要带上之前对话的历史。这里是使用一个message数组来存储对话历史，每一次对话都会把message的内容落盘。

agent可以调用工具，这些工具最基本的是读写文件的能力。每一轮对话中，agent输出 tool use的时候，json解析之后调用工具。然后把工具调用的结果塞回message中。这里真正给大模型加上了手脚。

对话历史不能无限增长，所以会有一定的压缩机制。包括对于工具调用的结果进行截断压缩，多次读同一个文件的工具调用结果进行删除。当达到一定阈值，会调用模型对message进行一次summary。

单就压缩message这件事情，应该会有许多方法。也可以看出agent本质上是一个工程，是一个系统。

然后就是memory，memory是一些记忆的点，你主动希望让模型记住，能够emphasize的点，而不只是存在对话历史中。memory同样需要落盘，MEMORY.md 是一堆memory文件的一个索引，每次构建systemprompt的时候传一个索引，agent可以根据自己的需要去读对应的memory文件。

skill是一个可插拔的组件，本质上是prompt，会落盘到文件系统中。每次调用的时候提供每个skill的简单描述，由模型决定使用哪一个skill。

MCP是agent调用服务的一种通信协议，我们主要实现的正是一种通信协议，如何建立连接，通过一个信箱通信。对应的MCP服务器可能有github服务器，browser服务器，通过MCP通信调用外部的服务器，可以实现更加丰富的功能。实际上MCP本质上就是一种tool。只不过是通过通信外部调用的tool。

