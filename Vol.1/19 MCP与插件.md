# MCP与插件

## 名词解释

`MCP client`：负责启动外部进程、发送请求、接收响应；

`MCP server`：MCP工具提供者，对外暴露一组能力；
https://blog.csdn.net/fufan_LLM/article/details/146510133

`MCP tool`：真正调用的工具；

`plugin`：告诉系统要发现和启动哪些 server；


## 最小心智模型

### MCP

```
LLM
  |
  | asks to call a tool
  v
Agent tool router
  |
  +-- native tool  -> 本地 Python handler
  |
  +-- MCP tool     -> 外部 MCP server
                        |
                        v
                    return result
```

