#!/usr/bin/env bash
# Manage the resident SkinTokens bpy service used by development workflows.
#
# This helper authenticates one exact service generation.  A plain /ping
# response is never sufficient: state, /proc identity, runtime environment,
# the authenticated health document, and the actual listening socket must all
# agree before READY is reported.
#
#   tools/dev_warm_services.sh start|stop|status|ready
#
# The tracked SkinTokens demo still starts its own bpy child.  Callers must not
# claim warm reuse until they use an explicit consumer-side reuse launcher.
set -euo pipefail
umask 077

readonly SERVICE_SCHEMA=avengine_tokenrig_warm_state_v1
readonly HEALTH_SCHEMA=avengine_tokenrig_warm_health_v1
readonly BIND_HOST=127.0.0.1
readonly HYGIENE_SHA=5052f0a1837011bf41a4ee20acb3c99ad5df2df8af99066bcd893f971604d8c2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SPEAR_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
SKT_INPUT=${AVENGINE_SKINTOKENS_ROOT:-$SPEAR_ROOT/../SkinTokens}
PATCH=$SPEAR_ROOT/tools/runtime_patches/fixed_skeleton_skintokens
WARM_ROOT=${AVENGINE_DEV_WARM_ROOT:-$SPEAR_ROOT/tmp/dev_warm_services}
TOKENRIG_PORT=${AVENGINE_WARM_TOKENRIG_PORT:-47652}
TOKENRIG_GPU=${AVENGINE_WARM_TOKENRIG_GPU:-3}

if [[ ! -d $SKT_INPUT ]]; then
  echo "tokenrig: SkinTokens root is not a directory: $SKT_INPUT" >&2
  exit 2
fi
SKT=$(cd -- "$SKT_INPUT" && pwd -P)
if [[ ! $TOKENRIG_PORT =~ ^[0-9]+$ ]] \
  || (( 10#$TOKENRIG_PORT < 1024 || 10#$TOKENRIG_PORT > 65535 )); then
  echo "tokenrig: invalid port: $TOKENRIG_PORT" >&2
  exit 2
fi
if [[ ! $TOKENRIG_GPU =~ ^[0-9]+$ ]] || (( 10#$TOKENRIG_GPU > 63 )); then
  echo "tokenrig: invalid GPU index: $TOKENRIG_GPU" >&2
  exit 2
fi
TOKENRIG_PORT=$((10#$TOKENRIG_PORT))
TOKENRIG_GPU=$((10#$TOKENRIG_GPU))
if [[ $WARM_ROOT != /* ]]; then
  echo "tokenrig: AVENGINE_DEV_WARM_ROOT must be absolute" >&2
  exit 2
fi

TOKENRIG_DIR=$WARM_ROOT/tokenrig
STATEFILE=$TOKENRIG_DIR/server.state
LOCKFILE=$TOKENRIG_DIR/service.lock
LEGACY_PIDFILE=$TOKENRIG_DIR/server.pid

verify_dependencies() {
  local command
  for command in curl flock sha256sum ss; do
    command -v "$command" >/dev/null 2>&1 || {
      echo "tokenrig: required command is unavailable: $command" >&2
      return 2
    }
  done
}

verify_tokenrig_installation() {
  local actual_hygiene_sha
  [[ -x $SKT/.venv/bin/python && -f $SKT/bpy_server.py ]] || {
    echo "tokenrig: SkinTokens runtime is incomplete: $SKT" >&2
    return 2
  }
  [[ -f $PATCH/sitecustomize.py && ! -L $PATCH/sitecustomize.py ]] || {
    echo "tokenrig: hygiene patch is not a direct file: $PATCH/sitecustomize.py" >&2
    return 2
  }
  actual_hygiene_sha=$(sha256sum "$PATCH/sitecustomize.py" | cut -d' ' -f1)
  [[ $actual_hygiene_sha == "$HYGIENE_SHA" ]] || {
    echo "tokenrig: hygiene patch hash changed: $actual_hygiene_sha" >&2
    return 2
  }
}

secure_state_directories() {
  local path owner
  for path in "$WARM_ROOT" "$TOKENRIG_DIR"; do
    if [[ -L $path || (-e $path && ! -d $path) ]]; then
      echo "tokenrig: state path must be a direct directory: $path" >&2
      return 2
    fi
  done
  mkdir -p -- "$TOKENRIG_DIR"
  for path in "$WARM_ROOT" "$TOKENRIG_DIR"; do
    owner=$(stat -c '%u' "$path") || return 2
    if [[ $owner != "$EUID" ]]; then
      echo "tokenrig: state directory is not owned by the caller: $path" >&2
      return 2
    fi
  done
  chmod 700 -- "$WARM_ROOT" "$TOKENRIG_DIR"
}

prepare_lock_file() {
  local owner mode links
  if [[ ! -e $LOCKFILE ]]; then
    : > "$LOCKFILE"
  fi
  [[ -f $LOCKFILE && ! -L $LOCKFILE ]] || {
    echo "tokenrig: lock path is not a direct file: $LOCKFILE" >&2
    return 2
  }
  owner=$(stat -c '%u' "$LOCKFILE") || return 2
  mode=$(stat -c '%a' "$LOCKFILE") || return 2
  links=$(stat -c '%h' "$LOCKFILE") || return 2
  [[ $owner == "$EUID" && $links == 1 ]] || {
    echo "tokenrig: lock file identity is unsafe: $LOCKFILE" >&2
    return 2
  }
  [[ $mode == 600 ]] || chmod 600 -- "$LOCKFILE"
}

lock_exclusive() {
  secure_state_directories
  prepare_lock_file
  exec 9<>"$LOCKFILE"
  flock -x 9
}

lock_shared_if_present() {
  [[ -d $TOKENRIG_DIR && ! -L $TOKENRIG_DIR ]] || return 1
  secure_state_directories
  prepare_lock_file
  exec 9<>"$LOCKFILE"
  flock -s 9
}

port_has_listener() {
  ss -H -ltn "sport = :$TOKENRIG_PORT" 2>/dev/null | grep -q .
}

new_generation() {
  local generation
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    generation=$(tr -d -- '-' < /proc/sys/kernel/random/uuid)
  else
    generation=$(od -An -N16 -tx1 /dev/urandom | tr -d '[:space:]')
  fi
  [[ $generation =~ ^[0-9a-f]{32}$ ]] || {
    echo "tokenrig: could not create a service generation" >&2
    return 2
  }
  printf '%s\n' "$generation"
}

proc_starttime() {
  local pid=$1
  [[ -r /proc/$pid/stat ]] || return 1
  awk '{print $22}' "/proc/$pid/stat"
}

reset_loaded_state() {
  STATE_SCHEMA=
  STATE_GENERATION=
  STATE_PID=
  STATE_STARTTIME=
  STATE_PATCH_SHA=
  STATE_BIND_HOST=
  STATE_PORT=
  STATE_GPU=
  STATE_PYTHON=
  STATE_SKT_ROOT=
  STATE_GENERATION_DIR=
  STATE_LOG_PATH=
  STATE_AUDIT_PATH=
  STATE_MARKER_DIR=
}

load_state() {
  local key value owner mode links
  reset_loaded_state
  [[ -f $STATEFILE && ! -L $STATEFILE ]] || return 1
  owner=$(stat -c '%u' "$STATEFILE") || return 1
  mode=$(stat -c '%a' "$STATEFILE") || return 1
  links=$(stat -c '%h' "$STATEFILE") || return 1
  [[ $owner == "$EUID" && $mode == 600 && $links == 1 ]] || return 1
  while IFS='=' read -r key value || [[ -n ${key:-}${value:-} ]]; do
    case "$key" in
      schema) [[ -z $STATE_SCHEMA ]] || return 1; STATE_SCHEMA=$value ;;
      generation) [[ -z $STATE_GENERATION ]] || return 1; STATE_GENERATION=$value ;;
      pid) [[ -z $STATE_PID ]] || return 1; STATE_PID=$value ;;
      starttime) [[ -z $STATE_STARTTIME ]] || return 1; STATE_STARTTIME=$value ;;
      patch_sha256) [[ -z $STATE_PATCH_SHA ]] || return 1; STATE_PATCH_SHA=$value ;;
      bind_host) [[ -z $STATE_BIND_HOST ]] || return 1; STATE_BIND_HOST=$value ;;
      port) [[ -z $STATE_PORT ]] || return 1; STATE_PORT=$value ;;
      gpu) [[ -z $STATE_GPU ]] || return 1; STATE_GPU=$value ;;
      python) [[ -z $STATE_PYTHON ]] || return 1; STATE_PYTHON=$value ;;
      skintokens_root) [[ -z $STATE_SKT_ROOT ]] || return 1; STATE_SKT_ROOT=$value ;;
      generation_dir) [[ -z $STATE_GENERATION_DIR ]] || return 1; STATE_GENERATION_DIR=$value ;;
      log_path) [[ -z $STATE_LOG_PATH ]] || return 1; STATE_LOG_PATH=$value ;;
      audit_path) [[ -z $STATE_AUDIT_PATH ]] || return 1; STATE_AUDIT_PATH=$value ;;
      marker_dir) [[ -z $STATE_MARKER_DIR ]] || return 1; STATE_MARKER_DIR=$value ;;
      *) return 1 ;;
    esac
  done < "$STATEFILE"

  [[ $STATE_SCHEMA == "$SERVICE_SCHEMA" ]] || return 1
  [[ $STATE_GENERATION =~ ^[0-9a-f]{32}$ ]] || return 1
  [[ $STATE_PID =~ ^[0-9]+$ ]] && (( 10#$STATE_PID > 1 )) || return 1
  [[ $STATE_STARTTIME =~ ^[0-9]+$ ]] || return 1
  [[ $STATE_PATCH_SHA =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ $STATE_BIND_HOST == "$BIND_HOST" ]] || return 1
  [[ $STATE_PORT =~ ^[0-9]+$ ]] \
    && (( 10#$STATE_PORT >= 1024 && 10#$STATE_PORT <= 65535 )) || return 1
  [[ $STATE_GPU =~ ^[0-9]+$ ]] && (( 10#$STATE_GPU <= 63 )) || return 1
  [[ $STATE_GENERATION_DIR == "$TOKENRIG_DIR/generations/$STATE_GENERATION" ]] || return 1
  [[ $STATE_LOG_PATH == "$STATE_GENERATION_DIR/server.log" ]] || return 1
  [[ $STATE_AUDIT_PATH == "$STATE_GENERATION_DIR/load_audit.jsonl" ]] || return 1
  [[ $STATE_MARKER_DIR == "$STATE_GENERATION_DIR/runtime_markers" ]] || return 1
  [[ $STATE_PYTHON == "$STATE_SKT_ROOT/.venv/bin/python" ]] || return 1
}

state_matches_current_configuration() {
  [[ $STATE_PATCH_SHA == "$HYGIENE_SHA" \
    && $STATE_BIND_HOST == "$BIND_HOST" \
    && $STATE_PORT == "$TOKENRIG_PORT" \
    && $STATE_GPU == "$TOKENRIG_GPU" \
    && $STATE_PYTHON == "$SKT/.venv/bin/python" \
    && $STATE_SKT_ROOT == "$SKT" ]]
}

proc_environment_contains() {
  local pid=$1
  local expected=$2
  local item
  [[ -r /proc/$pid/environ ]] || return 1
  while IFS= read -r -d '' item; do
    [[ $item == "$expected" ]] && return 0
  done < "/proc/$pid/environ"
  return 1
}

process_matches_loaded_state() {
  local actual_starttime actual_exe expected_exe actual_cwd
  local -a arguments=()
  kill -0 "$STATE_PID" 2>/dev/null || return 1
  actual_starttime=$(proc_starttime "$STATE_PID") || return 1
  [[ $actual_starttime == "$STATE_STARTTIME" ]] || return 1
  actual_exe=$(readlink -f "/proc/$STATE_PID/exe") || return 1
  expected_exe=$(readlink -f "$STATE_PYTHON") || return 1
  [[ $actual_exe == "$expected_exe" ]] || return 1
  actual_cwd=$(readlink -f "/proc/$STATE_PID/cwd") || return 1
  [[ $actual_cwd == "$STATE_SKT_ROOT" ]] || return 1
  mapfile -d '' -t arguments < "/proc/$STATE_PID/cmdline"
  [[ ${#arguments[@]} == 3 \
    && ${arguments[0]} == "$STATE_PYTHON" \
    && ${arguments[1]} == -u \
    && ${arguments[2]} == bpy_server.py ]] || return 1

  proc_environment_contains "$STATE_PID" "TOKENRIG_SERVICE_GENERATION=$STATE_GENERATION" \
    && proc_environment_contains "$STATE_PID" "TOKENRIG_SERVER_HYGIENE_SHA256=$STATE_PATCH_SHA" \
    && proc_environment_contains "$STATE_PID" "TOKENRIG_BPY_BIND_HOST=$STATE_BIND_HOST" \
    && proc_environment_contains "$STATE_PID" "TOKENRIG_BPY_PORT=$STATE_PORT" \
    && proc_environment_contains "$STATE_PID" "CUDA_VISIBLE_DEVICES=$STATE_GPU" \
    && proc_environment_contains "$STATE_PID" "TOKENRIG_LOAD_AUDIT_PATH=$STATE_AUDIT_PATH" \
    && proc_environment_contains "$STATE_PID" "TOKENRIG_HYGIENE_MARKER_DIR=$STATE_MARKER_DIR"
}

expected_health_body() {
  printf \
    '{"bind_host":"%s","generation":"%s","patch_sha256":"%s","pid":%s,"port":%s,"schema":"%s"}' \
    "$STATE_BIND_HOST" "$STATE_GENERATION" "$STATE_PATCH_SHA" \
    "$STATE_PID" "$STATE_PORT" "$HEALTH_SCHEMA"
}

health_matches_loaded_state() {
  local actual expected
  expected=$(expected_health_body)
  actual=$(curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 \
    "http://$STATE_BIND_HOST:$STATE_PORT/avengine-health" 2>/dev/null) || return 1
  [[ $actual == "$expected" ]]
}

socket_matches_loaded_state() {
  local line
  local count=0
  while IFS= read -r line; do
    [[ -n $line ]] || continue
    count=$((count + 1))
    [[ $line == *"$STATE_BIND_HOST:$STATE_PORT"* \
      && $line == *"pid=$STATE_PID,"* ]] || return 1
  done < <(ss -H -ltnp "sport = :$STATE_PORT" 2>/dev/null)
  (( count == 1 ))
}

write_state_atomic() {
  local temporary
  temporary=$(mktemp "$TOKENRIG_DIR/.server.state.XXXXXX")
  chmod 600 -- "$temporary"
  {
    printf 'schema=%s\n' "$SERVICE_SCHEMA"
    printf 'generation=%s\n' "$STATE_GENERATION"
    printf 'pid=%s\n' "$STATE_PID"
    printf 'starttime=%s\n' "$STATE_STARTTIME"
    printf 'patch_sha256=%s\n' "$STATE_PATCH_SHA"
    printf 'bind_host=%s\n' "$STATE_BIND_HOST"
    printf 'port=%s\n' "$STATE_PORT"
    printf 'gpu=%s\n' "$STATE_GPU"
    printf 'python=%s\n' "$STATE_PYTHON"
    printf 'skintokens_root=%s\n' "$STATE_SKT_ROOT"
    printf 'generation_dir=%s\n' "$STATE_GENERATION_DIR"
    printf 'log_path=%s\n' "$STATE_LOG_PATH"
    printf 'audit_path=%s\n' "$STATE_AUDIT_PATH"
    printf 'marker_dir=%s\n' "$STATE_MARKER_DIR"
  } > "$temporary"
  sync -f "$temporary"
  mv -f -- "$temporary" "$STATEFILE"
  chmod 600 -- "$STATEFILE"
  sync -f "$TOKENRIG_DIR"
}

archive_loaded_state() {
  local classification=$1
  local destination=$STATE_GENERATION_DIR/state.$classification
  if [[ -f $STATEFILE && -d $STATE_GENERATION_DIR && ! -e $destination ]]; then
    mv -- "$STATEFILE" "$destination"
    chmod 600 -- "$destination"
  else
    rm -f -- "$STATEFILE"
  fi
}

legacy_pid() {
  local value
  [[ -f $LEGACY_PIDFILE && ! -L $LEGACY_PIDFILE ]] || return 1
  IFS= read -r value < "$LEGACY_PIDFILE" || return 1
  [[ $value =~ ^[0-9]+$ ]] && (( 10#$value > 1 )) || return 1
  printf '%s\n' "$value"
}

discard_dead_legacy_pidfile() {
  local pid
  [[ -e $LEGACY_PIDFILE ]] || return 0
  if pid=$(legacy_pid) && kill -0 "$pid" 2>/dev/null; then
    echo "tokenrig: legacy unmanaged process is still alive: pid=$pid" >&2
    return 3
  fi
  rm -f -- "$LEGACY_PIDFILE"
}

report_failure_log() {
  local log_path=$1
  echo "tokenrig: generation log: $log_path" >&2
  if [[ -f $log_path ]]; then
    tail -n 20 "$log_path" >&2 || true
  fi
}

status_locked() {
  if [[ ! -e $STATEFILE ]]; then
    if port_has_listener; then
      echo "tokenrig: STALE/EXTERNAL listener on $BIND_HOST:$TOKENRIG_PORT without authenticated state"
      return 3
    fi
    echo "tokenrig: DOWN"
    return 1
  fi
  if ! load_state; then
    echo "tokenrig: STALE invalid or insecure state file: $STATEFILE"
    return 3
  fi
  if ! process_matches_loaded_state; then
    echo "tokenrig: STALE generation=$STATE_GENERATION pid=$STATE_PID is not the recorded process"
    return 3
  fi
  if ! state_matches_current_configuration; then
    echo "tokenrig: STALE generation=$STATE_GENERATION pid=$STATE_PID configuration changed"
    return 3
  fi
  if health_matches_loaded_state && socket_matches_loaded_state; then
    echo "tokenrig: READY generation=$STATE_GENERATION pid=$STATE_PID host=$STATE_BIND_HOST port=$STATE_PORT"
    return 0
  fi
  if port_has_listener; then
    echo "tokenrig: STALE generation=$STATE_GENERATION pid=$STATE_PID listener identity mismatch"
    return 3
  fi
  echo "tokenrig: STARTING generation=$STATE_GENERATION pid=$STATE_PID log=$STATE_LOG_PATH"
  return 2
}

start_locked() {
  local generation pid starttime returncode
  local generation_dir log_path audit_path marker_dir
  local -a tokenrig_environment

  verify_dependencies
  verify_tokenrig_installation

  if [[ -e $STATEFILE ]]; then
    if ! load_state; then
      echo "tokenrig: refusing to replace invalid or insecure state: $STATEFILE" >&2
      return 3
    fi
    if process_matches_loaded_state; then
      if ! state_matches_current_configuration; then
        echo "tokenrig: authenticated generation is stale; stop it before restart: $STATE_GENERATION" >&2
        return 3
      fi
      if health_matches_loaded_state && socket_matches_loaded_state; then
        echo "tokenrig: already READY generation=$STATE_GENERATION pid=$STATE_PID"
        return 0
      fi
      if port_has_listener; then
        echo "tokenrig: listener does not authenticate as generation=$STATE_GENERATION" >&2
        return 3
      fi
      echo "tokenrig: already STARTING generation=$STATE_GENERATION pid=$STATE_PID"
      return 0
    fi
    if port_has_listener; then
      echo "tokenrig: dead/mismatched state coexists with a listener; refusing to launch" >&2
      return 3
    fi
    report_failure_log "$STATE_LOG_PATH"
    archive_loaded_state failed
  fi

  discard_dead_legacy_pidfile
  if port_has_listener; then
    echo "tokenrig: port already has an unauthenticated listener: $BIND_HOST:$TOKENRIG_PORT" >&2
    return 3
  fi

  generation=$(new_generation)
  generation_dir=$TOKENRIG_DIR/generations/$generation
  log_path=$generation_dir/server.log
  audit_path=$generation_dir/load_audit.jsonl
  marker_dir=$generation_dir/runtime_markers
  install -d -m 700 -- "$TOKENRIG_DIR/generations" "$generation_dir" "$marker_dir"
  : > "$log_path"
  chmod 600 -- "$log_path"

  tokenrig_environment=(
    "CUDA_VISIBLE_DEVICES=$TOKENRIG_GPU"
    "TOKENRIG_CANARY_SEED=42"
    "TOKENRIG_SERVICE_GENERATION=$generation"
    "TOKENRIG_LOAD_AUDIT_PATH=$audit_path"
    "TOKENRIG_HYGIENE_MARKER_DIR=$marker_dir"
    "TOKENRIG_SERVER_HYGIENE_SHA256=$HYGIENE_SHA"
    "TOKENRIG_BPY_BIND_HOST=$BIND_HOST"
    "TOKENRIG_BPY_PORT=$TOKENRIG_PORT"
    "PYTHONPATH=$PATCH:$SKT"
    "MPLCONFIGDIR=$generation_dir/matplotlib"
    "NUMBA_CACHE_DIR=$generation_dir/numba"
  )
  install -d -m 700 -- "$generation_dir/matplotlib" "$generation_dir/numba"

  (
    cd "$SKT"
    exec 9>&-
    nohup env "${tokenrig_environment[@]}" \
      "$SKT/.venv/bin/python" -u bpy_server.py \
      >> "$log_path" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$generation_dir/launcher.pid"
  )
  pid=$(<"$generation_dir/launcher.pid")
  rm -f -- "$generation_dir/launcher.pid"
  [[ $pid =~ ^[0-9]+$ ]] || {
    echo "tokenrig: launcher did not return a PID" >&2
    report_failure_log "$log_path"
    return 1
  }

  starttime=
  for _attempt in {1..20}; do
    starttime=$(proc_starttime "$pid" 2>/dev/null || true)
    [[ -n $starttime ]] && break
    sleep 0.05
  done
  if [[ -z $starttime ]]; then
    if wait "$pid" 2>/dev/null; then returncode=0; else returncode=$?; fi
    echo "tokenrig: process exited before identity publication: pid=$pid rc=$returncode" >&2
    report_failure_log "$log_path"
    return 1
  fi

  reset_loaded_state
  STATE_SCHEMA=$SERVICE_SCHEMA
  STATE_GENERATION=$generation
  STATE_PID=$pid
  STATE_STARTTIME=$starttime
  STATE_PATCH_SHA=$HYGIENE_SHA
  STATE_BIND_HOST=$BIND_HOST
  STATE_PORT=$TOKENRIG_PORT
  STATE_GPU=$TOKENRIG_GPU
  STATE_PYTHON=$SKT/.venv/bin/python
  STATE_SKT_ROOT=$SKT
  STATE_GENERATION_DIR=$generation_dir
  STATE_LOG_PATH=$log_path
  STATE_AUDIT_PATH=$audit_path
  STATE_MARKER_DIR=$marker_dir
  write_state_atomic

  for _attempt in {1..10}; do
    if ! process_matches_loaded_state; then
      echo "tokenrig: process exited or changed identity immediately after launch: pid=$pid" >&2
      archive_loaded_state failed
      report_failure_log "$log_path"
      return 1
    fi
    health_matches_loaded_state && socket_matches_loaded_state && break
    sleep 0.05
  done
  if health_matches_loaded_state && socket_matches_loaded_state; then
    echo "tokenrig: READY generation=$generation pid=$pid host=$BIND_HOST port=$TOKENRIG_PORT"
  else
    echo "tokenrig: STARTING generation=$generation pid=$pid host=$BIND_HOST port=$TOKENRIG_PORT"
    echo "tokenrig: poll readiness with: $0 ready"
  fi
}

stop_locked() {
  local pid generation
  if [[ ! -e $STATEFILE ]]; then
    if port_has_listener; then
      echo "tokenrig: listener has no authenticated state; refusing to stop" >&2
      return 3
    fi
    discard_dead_legacy_pidfile
    echo "tokenrig: not running"
    return 0
  fi
  if ! load_state; then
    echo "tokenrig: invalid state; refusing to signal any PID: $STATEFILE" >&2
    return 3
  fi
  if ! process_matches_loaded_state; then
    echo "tokenrig: state does not authenticate pid=$STATE_PID; refusing to signal it" >&2
    return 3
  fi
  pid=$STATE_PID
  generation=$STATE_GENERATION
  kill -TERM "$pid"
  for _attempt in {1..10}; do
    process_matches_loaded_state || break
    sleep 1
  done
  if process_matches_loaded_state; then
    echo "tokenrig: generation did not stop after 10s: generation=$generation pid=$pid" >&2
    return 1
  fi
  archive_loaded_state stopped
  rm -f -- "$LEGACY_PIDFILE"
  echo "tokenrig: stopped generation=$generation pid=$pid"
}

verify_dependencies
action=${1:-status}
case "$action" in
  start)
    lock_exclusive
    start_locked
    ;;
  stop)
    lock_exclusive
    stop_locked
    ;;
  status|ready)
    if lock_shared_if_present; then
      status_locked
    elif port_has_listener; then
      echo "tokenrig: STALE/EXTERNAL listener on $BIND_HOST:$TOKENRIG_PORT without state directory"
      exit 3
    else
      echo "tokenrig: DOWN"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 start|stop|status|ready" >&2
    exit 2
    ;;
esac
