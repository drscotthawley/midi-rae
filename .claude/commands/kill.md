Kill a process on a remote host by PID.

Usage: /kill <host> <pid>

Parse $ARGUMENTS as "<host> <pid>" and run from `/workspaces/ClaudeCode-Mar12/midi-rae`:
`bash scripts/kill.sh <host> <pid>`

Then confirm by checking status with `bash scripts/status.sh <host>`.

Reminder: use `razer-docker` (home) or `razer-ts-docker` (Tailscale) for razer, `lecun` for lecun.
