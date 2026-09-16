# 请求分发服务器

请求分发服务可以将多个 api_server 服务，进行并联。用户可以只需要访问代理 URL，就可以间接访问不同的 api_server 服务。代理服务内部会自动分发请求，做到负载均衡。

## 启动

启动代理服务：

```shell
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "min_expected_latency" --serving-strategy Hybrid
```

启动成功后，代理服务的 URL 也会被脚本打印。浏览器访问这个 URL，可以打开 Swagger UI。
随后，用户可以在启动 api_server 服务的时候，通过 `--proxy-url` 命令将其直接添加到代理服务中。例如：`lmdeploy serve api_server InternLM/internlm2-chat-1_8b --proxy-url http://0.0.0.0:8000`。
这样，用户可以通过代理节点访问 api_server 的服务，代理节点的使用方式和 api_server 一模一样，都是兼容 OpenAI 的形式。

- /v1/models
- /v1/chat/completions
- /v1/completions

## 节点管理

通过 Swagger UI，我们可以看到多个 API。其中，和 api_server 节点管理相关的有：

- /nodes/status
- /nodes/add
- /nodes/remove

他们分别表示，查看所有的 api_server 服务节点，增加某个节点，删除某个节点。他们的使用方式，最直接的可以在浏览器里面直接操作。也可以通过命令行或者 python 操作。

### 通过 command 增删查

```shell
curl -X 'GET' \
  'http://localhost:8000/nodes/status' \
  -H 'accept: application/json'
```

```shell
curl -X 'POST' \
  'http://localhost:8000/nodes/add' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "http://0.0.0.0:23333"
}'
```

```shell
curl -X 'POST' \
  'http://localhost:8000/nodes/remove?node_url=http://0.0.0.0:23333' \
  -H 'accept: application/json' \
  -d ''
```

### 通过 python 脚本增删查

```python
# 查询所有节点
import requests
url = 'http://localhost:8000/nodes/status'
headers = {'accept': 'application/json'}
response = requests.get(url, headers=headers)
print(response.text)
```

```python
# 添加新节点
import requests
url = 'http://localhost:8000/nodes/add'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {"url": "http://0.0.0.0:23333"}
response = requests.post(url, headers=headers, json=data)
print(response.text)
```

```python
# 删除某个节点
import requests
url = 'http://localhost:8000/nodes/remove'
headers = {'accept': 'application/json',}
params = {'node_url': 'http://0.0.0.0:23333',}
response = requests.post(url, headers=headers, data='', params=params)
print(response.text)
```

## 服务策略

LMDeploy 当前支持混合部署服务（Hybrid），以及 PD 分离部署服务（DistServe）

- Hybrid: 不区分 Prefill 和 Decoding 实例，即传统的推理部署模式。
- DistServe: 将 Prefill 和 Decoding 实例分离，部署在不同的服务节点上以实现更灵活高效的资源调度和扩展。

## 分发策略

代理服务目前的分发策略如下：

- random： 根据用户提供的各个 api_server 节点的处理请求的能力，进行有权重的随机。处理请求的吞吐量越大，就越有可能被分配。部分节点没有提供吞吐量，将按照其他节点的平均吞吐量对待。
- min_expected_latency： 根据每个节点现有的待处理完的请求，和各个节点吞吐能力，计算预期完成响应所需时间，时间最短的将被分配。未提供吞吐量的节点，同上。
- min_observed_latency： 根据每个节点过去一定数量的请求，处理完成所需的平均用时，用时最短的将被分配。

### DistServe 连接预热资源限制

`POST /distserve/connection_warmup` 同时只允许一轮预热；并发的重复请求返回 HTTP 409。
每轮最多使用 32 个 worker，不会按 Prefill × Decode 笛卡尔积一次性创建全部任务。
连接失败返回 HTTP 503，超时返回 HTTP 504；后端恢复后可重试，已成功建立的连接会复用。

启动 Proxy 前可设置以下环境变量：

| 变量                               | 默认值 | 含义                                                             |
| ---------------------------------- | ------ | ---------------------------------------------------------------- |
| `LMDEPLOY_PD_CONNECTION_TIMEOUT`   | `60`   | 单个连接对的总期限（秒），覆盖 HTTP 请求和重试；必须是有限正数。 |
| `LMDEPLOY_PD_WARMUP_TIMEOUT`       | `300`  | 一轮预热的总期限（秒）；必须是有限正数。                         |
| `LMDEPLOY_PD_MAX_CONNECT_REQUESTS` | `2048` | 同时等待建连的调用者上限；必须是正整数，超过后立即失败。         |

单连接对期限及建连调用者配额也适用于按需 PD 建连。同一连接对共享建连任务，单个调用者取消不会中断其他调用者；
最后一个调用者离开时，未完成的建连任务会被取消。Proxy 退出时会关闭建连任务及 HTTP 资源。
通用 `AIOHTTP_TIMEOUT` 不会关闭 PD 建连期限；大型或慢速拓扑应适当调高上述限制，而不是使用无限超时。

资源限制不等于身份认证。请通过可信管理网络或具备管理鉴权的上游网关限制 `/nodes/*` 和
`/distserve/*` 的访问；普通推理 API key 不代表独立的管理角色。

### 异步 KV 迁移

在 Prefill/Decode API Server 设置 `LMDEPLOY_USE_ASYNC_MIGRATION=1`，可将 Mooncake
阻塞传输（或 DLSlime 完成等待）放到工作线程执行。未设置或设为 `0` 时使用同步迁移。
传输失败仍向调用方抛出。取消异步迁移时，会等待原生操作结束后再返回；Python 取消
不能停止在途 RDMA，也不能提前允许其缓冲区被复用。这不等于增加原生传输超时或实现
分布式缓存垃圾回收。
