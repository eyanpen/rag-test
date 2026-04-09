# activate.sh
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "❌ 请使用 source activate.sh 运行"
  echo "source activate.sh "
  exit 1
fi

if [[ -f "venv/bin/activate" ]]; then
  source "venv/bin/activate"
elif [[ -f "venv/Scripts/activate" ]]; then
  source "venv/Scripts/activate"
else
  echo "❌ 未找到虚拟环境，请确认 venv 已创建"
  return 1
fi
export ERICAI_TOKEN_FILE_PATH="$HOME/.ericai_token"
export CONFIG_FILE_PATH="$HOME/.config/indexing-config.yaml"
# ============================
# 获取匹配的最年轻 pod
_get_youngest_pod() {
  local keyword="$1"
  local ns="$2"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  # 获取 pod 名称和创建时间，按创建时间排序
  pod=$(kubectl get pods $ns_opt -o json \
    | jq -r '.items[] | select(.metadata.name | contains("'"$keyword"'")) | "\(.metadata.name) \(.metadata.creationTimestamp)"' \
    | sort -k2 \
    | head -n1 \
    | awk '{print $1}')   # 只取 pod 名称

  if [ -z "$pod" ]; then
    echo "No pod matching '$keyword' found"
    return 1
  fi

  echo "$pod"
}

# ============================
# 切换 namespace
kubectl_setNamespace() {
  if [ -z "$1" ]; then
    kubectl config view --minify | grep namespace
    return 0
  fi
  local current
  current=$(kubectl config view --minify --output 'jsonpath={.contexts[0].context.namespace}' 2>/dev/null)
  if [ "$current" = "$1" ]; then
    echo "Already in namespace: $1"
    return 0
  fi
  if ! kubectl get ns "$1" >/dev/null 2>&1; then
    echo "Namespace '$1' does not exist"
    return 1
  fi
  kubectl config set-context --current --namespace="$1"
  echo "Switched to namespace: $1"
}
# 创建 namespace（已存在则跳过）
kubectl_createNamespace() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <namespace>"
    return 1
  fi
  if kubectl get ns "$1" >/dev/null 2>&1; then
    echo "Namespace '$1' already exists, skipping."
    return 0
  fi
  kubectl create namespace "$1"
}

# 在 kohn040 和 minikube 两个 kubeconfig 之间切换
kubectlswitch() {
  local kohn040="$HOME/.kube/kohn040.config"
  local minikube="$HOME/.kube/minikube.config"

  if [[ "$KUBECONFIG" == "$kohn040" ]]; then
    export KUBECONFIG="$minikube"
    kubectl_setNamespace data-pipeline
  else
    export KUBECONFIG="$kohn040"
    kubectl_setNamespace data-pipeline-eyanpen
  fi
  echo "KUBECONFIG switched to: $KUBECONFIG"
}

# ============================
# 查看 pods
kpods() {
  [ -n "$1" ] && ns_opt="-n $1" || ns_opt=""
  kubectl get pods $ns_opt -o wide
}

# ============================
# 查看 pod logs（关键词匹配最年轻 pod）
klog() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> [namespace]"
    return 1
  fi
  local keyword="$1"
  local ns="$2"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  kubectl logs -f $ns_opt "$pod"  --all-containers=true | less
}

# ============================
# exec 进入 pod
kexec() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> [namespace] [command]"
    return 1
  fi
  local keyword="$1"
  local ns="$2"
  local cmd="${3:-/bin/sh}"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  kubectl exec -it $ns_opt "$pod" -- $cmd
}

# ============================
# port-forward
kpf() {
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> <local_port>:<remote_port> [namespace]"
    return 1
  fi
  local keyword="$1"
  local ports="$2"
  local ns="$3"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  kubectl port-forward $ns_opt "$pod" "$ports"
}

# ============================
# 查看 services
ksvc() {
  [ -n "$1" ] && ns_opt="-n $1" || ns_opt=""
  kubectl get svc $ns_opt
}

# ============================
# 删除 pod
kdel() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> [namespace]"
    return 1
  fi
  local keyword="$1"
  local ns="$2"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  kubectl delete pod $ns_opt "$pod"
}

# ============================
# describe pod
kdesc() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> [namespace]"
    return 1
  fi
  local keyword="$1"
  local ns="$2"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  kubectl describe pod $ns_opt "$pod"
}

# ============================
# 查看 PVC
kpvc() {
  [ -n "$1" ] && ns_opt="-n $1" || ns_opt=""
  kubectl get pvc $ns_opt
}

# ============================
# 删除 PVC（关键词匹配或删除全部）
kdpvc() {
  if [ -z "$1" ]; then
    echo "Usage: ${FUNCNAME[0]} <pvc_name|all> [namespace]"
    echo "  kdpvc workdir          删除名称含 'workdir' 的 PVC"
    echo "  kdpvc all              删除所有 PVC"
    echo "  kdpvc all data-pipeline 删除指定 namespace 的所有 PVC"
    return 1
  fi
  local target="$1"
  local ns="$2"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  if [ "$target" = "all" ]; then
    echo "删除所有 PVC..."
    kubectl delete pvc --all $ns_opt
  else
    local pvcs=$(kubectl get pvc $ns_opt --no-headers 2>/dev/null | awk -v kw="$target" '$1 ~ kw {print $1}')
    if [ -z "$pvcs" ]; then
      echo "未找到匹配 '$target' 的 PVC"
      return 1
    fi
    echo "$pvcs" | while read -r pvc; do
      echo "删除 PVC: $pvc"
      kubectl delete pvc $ns_opt "$pvc"
    done
  fi
}
kjobs()
{
  echo "========="
  echo "CronJobs:"
  kubectl get cronjob -A
  echo 
  echo "========="
  echo "Jobs:"
  kubectl get job -A --sort-by=.status.startTime
  echo "========="
}


# ============================
# 从 pod 复制文件或文件夹到本地
kcp() {
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ${FUNCNAME[0]} <pod_keyword> <remote_path> [local_path] [namespace]"
    echo "  kcp indexing /tmp/graphrag-project ./output"
    echo "  kcp indexing /tmp/graphrag-project              # 复制到当前目录"
    echo "  kcp indexing /app/logs/app.log ./app.log data-pipeline"
    return 1
  fi
  local keyword="$1"
  local remote_path="$2"
  local local_path="${3:-.}"
  local ns="$4"
  local ns_opt=""
  [ -n "$ns" ] && ns_opt="-n $ns"

  local pod=$(_get_youngest_pod "$keyword" "$ns") || return 1
  echo "从 $pod:$remote_path 复制到 $local_path ..."
  kubectl cp $ns_opt "$pod:$remote_path" "$local_path"
}

alias submit=' ./argo-pipeline.sh submit -p indexing_mode="graphrag"'



localrungraph() {
  local default_root="/home/eyanpen/sourceCode/rnd-ai-engine-features/examples/data-pipeline/k8s-middle-data-for-index/tmp-dir"
  # 如果参数中已包含 --root，则不自动添加
  local has_root=false
  for arg in "$@"; do
    if [[ "$arg" == "--root" ]]; then
      has_root=true
      break
    fi
  done
  if $has_root; then
    python3 -m indexing.graphrag.cli "$@"
  else
    python3 -m indexing.graphrag.cli "$@" --root "$default_root"
  fi
}
refreshEricAiToken()
{
  local cmd="ericai --ericsson-access-token --ericsson-force-new-access-token > $ERICAI_TOKEN_FILE_PATH"
  echo "+ $cmd"
  eval "$cmd"
}
alias eristatus='ericai eristatus --pinned --loaded'

helmupgrade()
{
  local cmd="helm upgrade data-pipeline data-pipeline/helm/eric-ai-data-pipeline -f pipeline-deployments.values.yaml -n data-pipeline"
  echo "+ $cmd"
  eval "$cmd"
}
runUbuntu()
{
  kubectl run ubuntu --image=nicolaka/netshoot -- sleep infinity
}
loginUbuntu()
{
  kubectl exec -it ubuntu -- /bin/bash
}
calulateFiles()
{
  # 按文件类型统计
find . -type f | awk -F. '
NF>1 {ext[$NF]++} 
END {
  total=0
  for (e in ext) {total+=ext[e]; print e ":" ext[e]}
  print "Total files:" total
}'
}
echo "Virtualenv activated: $VIRTUAL_ENV"
