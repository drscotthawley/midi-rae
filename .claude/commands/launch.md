Launch a midi-rae training run on a remote host.

Run `bash scripts/launch.sh $ARGUMENTS` from `/workspaces/ClaudeCode-Mar12/midi-rae`.

Arg order: host, type, config_name, bare_tag, [++hydra.overrides...]
- host: lecun | razer-docker | razer-ts-docker
- type: enc | dec | flow | fitpca | preencode | generate
- config_name: e.g. config_swin
- bare_tag: short name prefix, e.g. coarse10 or dec39 (no "tag=" prefix)
- overrides: e.g. ++flow.mode=coarse ++flow.n_epochs=200

Always commit pending changes before launching (per feedback_commit_on_launch).
Never use --force on lecun.
