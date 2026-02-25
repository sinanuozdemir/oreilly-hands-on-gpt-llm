# Deploying on DigitalOcean Kubernetes (DOKS)

Deploy the vLLM + Gateway stack on a DigitalOcean Kubernetes cluster with GPU nodes.

## Prerequisites

- [doctl](https://docs.digitalocean.com/reference/doctl/how-to/install/) installed and authenticated (`doctl auth init`)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- [Docker](https://docs.docker.com/get-docker/) installed and running
- A DigitalOcean account with GPU access

## Available GPU Sizes

| Size Slug | GPU | VRAM | $/hr |
|---|---|---|---|
| `gpu-4000adax1-20gb` | RTX 4000 Ada | 20GB | $0.76 |
| `gpu-6000adax1-48gb` | RTX 6000 Ada | 48GB | $1.57 |
| `gpu-l40sx1-48gb` | L40S | 48GB | $1.57 |
| `gpu-h100x1-80gb` | H100 | 80GB | $3.39 |

The Qwen3-4B-AWQ model only needs ~3GB VRAM, so the cheapest available GPU is fine.

> **Note:** GPU availability varies by region. If one region is out of capacity, try another. NYC1, TOR1, and SFO3 are good starting points.

---

## Step 1: Create the Cluster

```bash
doctl kubernetes cluster create llm-cluster \
  --region tor1 \
  --version 1.34.1-do.3 \
  --node-pool "name=gateway;size=s-2vcpu-4gb;count=2" \
  --node-pool "name=gpu-worker;size=gpu-6000adax1-48gb;count=1" \
  --wait
```

If you get a "region has insufficient capacity" error, try a different region or GPU size:

```bash
# Try a different region
doctl kubernetes cluster create llm-cluster \
  --region nyc1 \
  --version 1.34.1-do.3 \
  --node-pool "name=gateway;size=s-2vcpu-4gb;count=2" \
  --node-pool "name=gpu-worker;size=gpu-4000adax1-20gb;count=1" \
  --wait
```

## Step 2: Configure kubectl

```bash
doctl kubernetes cluster kubeconfig save llm-cluster
# Verify nodes are ready (GPU node may take a couple of minutes)
kubectl get nodes
```

Wait until all nodes show `Ready`:

```
NAME               STATUS   ROLES    AGE   VERSION
gateway-xxxxx      Ready    <none>   2m    v1.34.1
gateway-yyyyy      Ready    <none>   2m    v1.34.1
gpu-worker-zzzzz   Ready    <none>   1m    v1.34.1
```

## Step 3: Create Container Registry and Push Gateway Image

The gateway needs a container image. DigitalOcean Container Registry (DOCR) is the easiest option.

```bash
# Create registry (free starter tier)
doctl registry create llm-registry --subscription-tier starter

# Login to registry
doctl registry login

# Build for linux/amd64 (required even if you're on Apple Silicon)
docker buildx build --platform linux/amd64 \
  -t registry.digitalocean.com/llm-registry/llm-gateway:latest \
  ./gateway/ \
  --push

# Integrate registry with cluster
doctl registry kubernetes-manifest | kubectl apply -f -
doctl kubernetes cluster registry add llm-cluster
```

## Step 4: Create Namespace and Secrets

```bash
kubectl apply -f k8s/00-namespace.yaml

kubectl create secret generic llm-secrets \
  -n llm-serving \
  --from-literal=api-key=your-secret-key-here \
  --from-literal=hf-token=your-hf-token-here
```

## Step 5: Deploy vLLM

```bash
kubectl apply -f k8s/02-vllm.yaml
```

The vLLM pod will pull a ~15GB image and then download the model (~3GB). Monitor progress:

```bash
# Watch pod status
kubectl get pods -n llm-serving -w

# Check vLLM logs (look for "Application startup complete")
kubectl logs -n llm-serving -l app=vllm -f
```

> **Important:** The K8s Service is named `vllm-svc` (not `vllm`) to avoid a conflict where Kubernetes injects a `VLLM_PORT` environment variable that clashes with vLLM's own config.

## Step 6: Deploy Gateway

```bash
kubectl apply -f k8s/03-gateway.yaml
```

## Step 7: Get External IP and Test

```bash
# Get the gateway's external IP
kubectl get svc -n llm-serving gateway
```

Wait for `EXTERNAL-IP` to be assigned (may take 1-2 minutes), then:

```bash
GATEWAY_IP=$(kubectl get svc gateway -n llm-serving -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$GATEWAY_IP/health

# Chat completion
curl http://$GATEWAY_IP/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-AWQ",
    "messages": [{"role": "user", "content": "What is LoRA fine-tuning? 2 sentences max."}],
    "max_tokens": 256
  }'
```

## Step 8: Use with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<GATEWAY_IP>/v1",
    api_key="your-secret-key-here",
)

# Non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen3-14B-AWQ",
    messages=[{"role": "user", "content": "Explain agentic AI in 3 bullet points."}],
    max_tokens=512,
)
print(response.choices[0].message.content)

# Streaming (plus disabled thinking)
stream = client.chat.completions.create(
    model="Qwen/Qwen3-14B-AWQ",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "Tell me a quick story"}
    ],
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Qwen3 Thinking Mode

Qwen3 models include a "thinking" mode that generates a `<think>...</think>` reasoning block before answering. This is great for complex tasks but wastes tokens on simple questions:

| Mode | Prompt | Time | Tokens |
|---|---|---|---|
| Thinking (default) | "what is 1+1?" | ~26s | 256 (hit limit, no answer) |
| `/no_think` | "what is 1+1?" | ~1s | 7 ("2.") |

To disable thinking, add `/no_think` to the system message:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-14B-AWQ",
    messages=[
        {"role": "system", "content": "You are a concise assistant. /no_think"},
        {"role": "user", "content": "What is LoRA fine-tuning?"},
    ],
    max_tokens=256,
)
```

## Swapping Models

Edit `k8s/02-vllm.yaml` and change the `--model` argument, then redeploy:

```bash
kubectl apply -f k8s/02-vllm.yaml
kubectl rollout restart deployment/vllm -n llm-serving
```

The model cache (PVC) persists across restarts, so previously downloaded models won't need re-downloading.

| Model | VRAM | Notes |
|---|---|---|
| `Qwen/Qwen3-4B-AWQ` | ~3 GB | Fast, good for demos |
| `Qwen/Qwen3-14B-AWQ` | ~9 GB | Best quality/speed balance |
| `Qwen/Qwen3-32B-AWQ` | ~18 GB | Highest quality, needs 24GB+ GPU |

---

## Tear Down

```bash
# Delete the cluster (stops all billing for nodes)
doctl kubernetes cluster delete llm-cluster --force

# Delete the container registry (optional)
doctl registry delete llm-registry --force
```

## Cost Summary

| Component | Size | $/hr |
|---|---|---|
| Gateway nodes (x2) | s-2vcpu-4gb | ~$0.03 |
| GPU node (x1) | gpu-6000adax1-48gb | ~$1.57 |
| **Total** | | **~$1.60/hr** |

## Troubleshooting

### GPU node stuck in NotReady
GPU nodes take longer to initialize (driver setup). Wait 2-3 minutes. If it doesn't resolve:
```bash
kubectl describe node <gpu-node-name>
```

### vLLM CrashLoopBackOff
Check logs for the root cause:
```bash
kubectl logs -n llm-serving -l app=vllm --tail=50
```
Common issue: if the K8s Service is named `vllm`, Kubernetes injects `VLLM_PORT=tcp://...` which crashes vLLM. The Service must be named something else (we use `vllm-svc`).

### Gateway ImagePullBackOff
- **"no match for platform"**: You built the image on ARM (Apple Silicon) but the cluster runs AMD64. Rebuild with `--platform linux/amd64`.
- **"unauthorized"**: Run `doctl registry kubernetes-manifest | kubectl apply -f -` to add registry credentials to the cluster.

### Region has insufficient capacity
Try a different region or GPU size. GPU availability changes frequently.
