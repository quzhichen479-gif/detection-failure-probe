# Security Policy

## Supported versions

The project is pre-1.0. Security fixes are applied to the latest commit on the default branch. A
formal version support window will be defined before the first stable release.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for the repository when available. If
it is not enabled, contact the maintainer through the public profile at
<https://github.com/quzhichen479-gif> and ask for a private reporting channel. Do not include exploit
details or private dataset content in a public issue.

Include the affected version/commit, platform, reproduction steps using synthetic data, impact,
and any suggested mitigation. Expect an acknowledgement target of seven days; this is a
maintainer goal, not a service-level guarantee.

## Current trust boundaries and attack surface

Detection Failure Probe processes untrusted-looking local content, but the person invoking it is
expected to control the dataset YAML and run directory.

- **YAML and labels:** YAML is parsed with `safe_load`; labels are numeric text. Referenced dataset
  paths must remain under the YAML directory. Large YAML, list, label, and prediction files have
  explicit limits. Pillow parses image metadata, so Pillow image decoders remain part of the attack
  surface.
- **Prediction JSON:** only JSON is accepted; non-standard NaN/Infinity constants are rejected.
  Pickle, checkpoints, Python configs, URLs, and archive extraction are not supported.
- **Outputs:** run names use a strict allowlist, existing directories are not reused, generated JSON
  and HTML use atomic replacement, and generated artifact symlinks are rejected before report/note
  writes. A local user who can modify a run concurrently may still race the process.
- **Review server:** binds only to loopback, requires a random per-process token for data/assets and
  writes, serves only image paths listed by the audit, applies CSP and related headers, and limits
  note requests. Other local processes and browser extensions operate within the same machine trust
  boundary. Stop the server when review is complete.
- **Availability:** deeply nested directories, extremely numerous small files, crafted compressed
  images, and very large but limit-compliant JSON may consume CPU, memory, or disk. The MVP does not
  provide hard process resource isolation.
- **Confidentiality:** no package code uploads data, but reports and notes remain plaintext on local
  disk. Anyone with filesystem access to the run or dataset can read them.

Do not run the tool with elevated privileges on content from an untrusted party. For hostile
datasets, use an OS-level sandbox with strict CPU, memory, and filesystem limits.
