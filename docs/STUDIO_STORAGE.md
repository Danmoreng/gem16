# Studio chat storage

Native Studio keeps `studio.db` and content-addressed `attachments/` under
`$XDG_DATA_HOME/gem16`, `~/.local/share/gem16`, or `%LOCALAPPDATA%\gem16`.
`GEM16_STUDIO_DATA_ROOT` overrides this directory for portable use and tests.
The database contains private conversation text, reasoning, system prompts,
request settings, model identities, and previous response attempts. It is not encrypted.
Do not put this directory in Git. Temporary chats do not write conversation content.

A single worker owns SQLite (pinned 3.53.4, FTS5 enabled, loadable extensions disabled).
WAL, full synchronous commits and atomic, flushed attachment writes protect checkpoints.
One process holds the data directory lock. Another Studio instance reports an error
instead of changing its running messages. Future schema versions fail closed.
A user message and pending answer are saved before HTTP dispatch; storage failure
prevents dispatch and exposes a retry. Streaming checkpoints run about once per second;
terminal events and normal shutdown also save. A crash can lose tokens after the most
recent checkpoint. Interrupted attempts remain readable and are excluded from replay.

Chats carry their profile, component repositories/revisions and paths. Custom paths
are explicitly marked unverified; they do not acquire the catalog's revision claim.
Each answer stores the request's generation/server settings; reopening restores the
system prompt, reasoning effort and output limit. Server capacity and model selection
stay explicit user choices. GPU sessions are disposable and are not stored in SQLite.
There is no automatic truncation or summarization when a context request fails.

Attachments are copied and deduplicated by SHA-256. Original files may be moved or
removed afterward. Loading verifies size/hash and rejects traversal/symlink references;
missing or corrupt media leaves the text readable but blocks continuation.
Bounds: 10,000 messages, 32 MiB text/metadata per conversation, 64 MiB per attachment,
256 MiB loaded attachment bytes per conversation. The work queue is bounded.

Validation: `cmake --build build/native-studio --parallel 4` and
`ctest --test-dir build/native-studio --output-on-failure`. Host tests cover persistence,
partial recovery, attempt/settings/media roundtrips, same-size corruption, missing media,
archive/pin/delete, duplicate-instance exclusion, schema rejection and transaction rollback.
Windows runtime and desktop interaction qualification remains separate.

## Search, export and backup

The sidebar uses SQLite FTS5 over titles, answers, reasoning, previous attempts,
document text and attachment names. Space-separated terms are literal word prefixes,
not executable FTS expressions. Search ignores case and Latin accents. Results carry
message positions and jump to the matching turn; at most 200 hits are returned.
The normal sidebar shows the latest 500 conversations, pinned first.

**Export JSON + Markdown** writes a new folder with `chat.json`, `chat.md` and
`attachments/`. It also works for temporary chats and unsaved text after a save error.
JSON retains settings, model identity, timestamps and attempts. Missing media is
identified in Markdown; it is never silently replaced.
**Back up all chats** uses SQLite's online backup API, checks the copied database,
and copies every referenced attachment after hash verification. An incomplete or
corrupt source fails the backup; only a successful bundle receives its final folder name.
Weights and app/server preferences are outside this conversation backup.

To restore, close Studio and preserve the existing data folder. Copy the backup's
`studio.db` and `attachments/` into an empty data folder (do not carry over old WAL/SHM
files), or launch Studio with `GEM16_STUDIO_DATA_ROOT` set to a copy of the backup.
No running database file should be overwritten. Schema migration builds the search
index transactionally; a newer schema remains protected from older Studio builds.

**Clean unused attachments** removes only hash-named regular files without any
message reference, including archived chats. Profile/model blobs are never touched.
Backup bundles own their attachment copies and remain independent of this cleanup.
Tests additionally cover search escaping/accent handling, search migration, export,
backup reopening, FTS deletion and reference-aware garbage collection.
