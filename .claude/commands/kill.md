Kill a process on a remote host by PID.

Usage: /kill <host> <pid>

Run `ssh -o ClearAllForwardings=yes $ARGUMENTS` — but the arguments are host and PID, so the actual command is:

Parse $ARGUMENTS as "<host> <pid>" and run:
`ssh -o ClearAllForwardings=yes <host> "kill <pid>"`

Then confirm by checking status with `bash scripts/status.sh <host>`.

Reminder: use `razer-docker` (home) or `razer-ts-docker` (Tailscale) for razer, `lecun` for lecun.
