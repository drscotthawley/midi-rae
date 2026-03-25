#!/bin/bash
# Kill a process on a remote host by PID.
# Usage: ./scripts/kill.sh <host> <pid>
HOST="${1:?Usage: $0 <host> <pid>}"
PID="${2:?Usage: $0 <host> <pid>}"
ssh -o ClearAllForwardings=yes "${HOST}" "kill ${PID}"
echo "Sent kill to PID ${PID} on ${HOST}"
