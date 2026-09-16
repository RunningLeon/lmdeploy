# Request Distributor Server

The request distributor service can parallelize multiple api_server services. Users only need to access the proxy URL, and they can indirectly access different api_server services. The proxy service will automatically distribute requests internally, achieving load balancing.

## Startup

Start the proxy service:

```shell
lmdeploy serve proxy --server-name {server_name} --server-port {server_port} --routing-strategy "min_expected_latency" --serving-strategy Hybrid
```

After startup is successful, the URL of the proxy service will also be printed by the script. Access this URL in your browser to open the Swagger UI.
Subsequently, users can add it directly to the proxy service when starting the `api_server` service by using the `--proxy-url` command. For example:
`lmdeploy serve api_server InternLM/internlm2-chat-1_8b --proxy-url http://0.0.0.0:8000`。
In this way, users can access the services of the `api_server` through the proxy node, and the usage of the proxy node is exactly the same as that of the `api_server`, both of which are compatible with the OpenAI format.

- /v1/models
- /v1/chat/completions
- /v1/completions

## Node Management

Through Swagger UI, we can see multiple APIs. Those related to api_server node management include:

- /nodes/status
- /nodes/add
- /nodes/remove

They respectively represent viewing all api_server service nodes, adding a certain node, and deleting a certain node.

### Node Management through curl

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

### Node Management through python

```python
# query all nodes
import requests
url = 'http://localhost:8000/nodes/status'
headers = {'accept': 'application/json'}
response = requests.get(url, headers=headers)
print(response.text)
```

```python
# add a new node
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
# delete a node
import requests
url = 'http://localhost:8000/nodes/remove'
headers = {'accept': 'application/json',}
params = {'node_url': 'http://0.0.0.0:23333',}
response = requests.post(url, headers=headers, data='', params=params)
print(response.text)
```

## Serving Strategy

LMDeploy currently supports two serving strategies:

- Hybrid: Does not distinguish between Prefill and Decoding instances, following the traditional inference deployment mode.
- DistServe: Separates Prefill and Decoding instances, deploying them on different service nodes to achieve more flexible and efficient resource scheduling and scalability.

## Dispatch Strategy

The current distribution strategies of the proxy service are as follows:

- random： dispatches based on the ability of each api_server node provided by the user to process requests. The greater the request throughput, the more likely it is to be allocated. Nodes that do not provide throughput are treated according to the average throughput of other nodes.
- min_expected_latency： allocates based on the number of requests currently waiting to be processed on each node, and the throughput capability of each node, calculating the expected time required to complete the response. The shortest one gets allocated. Nodes that do not provide throughput are treated similarly.
- min_observed_latency： allocates based on the average time required to handle a certain number of past requests on each node. The one with the shortest time gets allocated.

### DistServe connection warmup limits

`POST /distserve/connection_warmup` permits one warmup round at a time per proxy process; concurrent
rounds return HTTP 409. It uses at most 32 workers rather than creating one task
per Prefill × Decode pair. Connection failures return HTTP 503 and timeouts return
HTTP 504; retry after the backend becomes available. Successfully established
pairs are reused on subsequent rounds.

The following environment variables must be set before starting the proxy:

| Variable                           | Default | Meaning                                                                                              |
| ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `LMDEPLOY_PD_CONNECTION_TIMEOUT`   | `60`    | Finite positive total seconds per pair, including HTTP calls and retries.                            |
| `LMDEPLOY_PD_WARMUP_TIMEOUT`       | `300`   | Finite positive total seconds for one warmup round.                                                  |
| `LMDEPLOY_PD_MAX_CONNECT_REQUESTS` | `2048`  | Positive integer maximum number of callers waiting for connections; excess callers fail immediately. |

The pair deadline and connection-admission limit also apply to lazy PD connection
establishment. Calls for the same pair share a handshake; cancelling one caller does not cancel other callers, but
the last caller leaving cancels its unfinished handshake. The proxy closes
connection tasks and HTTP resources on shutdown. The general `AIOHTTP_TIMEOUT`
setting does not disable the PD connection deadline. Increase these limits for
large or slow topologies rather than using an unlimited timeout.

These resource limits are not authentication. Restrict `/nodes/*` and
`/distserve/*` to a trusted management network or an authenticated management
upstream; inference API keys do not provide a separate management role.

### Asynchronous KV migration

Set `LMDEPLOY_USE_ASYNC_MIGRATION=1` on the Prefill/Decode API servers to offload
blocking Mooncake transfers (or DLSlime completion waits) to a worker thread.
The default, or `0`, uses synchronous migration. Transfer failures still
propagate to the caller. Cancelling an asynchronous migration waits for the
native operation to finish before returning: Python cancellation cannot stop
in-flight RDMA or make its buffers safe to reuse. This does not impose a
native transfer timeout or implement distributed cache garbage collection.
