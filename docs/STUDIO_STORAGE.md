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
The real Studio lifecycle, Markdown renderer and host fixtures run as three separate
CTest processes. Linux test executables cap virtual memory at 2 GiB and each CTest has
a 60-second timeout. Windows runtime and interactive desktop qualification remains separate.

## Canvas revisions

Schema 3 adds chat-owned `canvases` and immutable `canvas_revisions` tables.
Document IDs are scoped to their conversation. An edit requires the current
revision and exactly one matching source fragment; rejected edits change nothing.
Canvas mutations and their tool transcript are committed before model continuation.
Restoring history appends a revision. Deleting a conversation cascades to its Canvas.
Limits are 16 documents, 128 revisions per document, 1 MiB per source revision and
32 MiB of Canvas source history per chat. Canvas screenshots are content-addressed image attachments on their tool-result
messages. They are saved, reloaded and resent as actual images, so the same model
can interpret earlier screenshots in context. Browser diagnostics remain text
in the tool result. Temporary chats retain screenshots only in memory. Incomplete/interrupted
tool exchanges are excluded from subsequent model context. Backup includes all
revisions; JSON export includes source history. Canvas source has no separate FTS
index; tool messages participate in ordinary chat indexing.

## Search, export and backup

The chat UI exposes search and automatic persistence. Manual title editing, pinning,
archiving, export, backup and attachment cleanup controls were removed from the chat
window at the owner's request. The storage operations below remain internal APIs;
they are not Studio buttons. Existing titles and pin/archive metadata are preserved.

The sidebar uses SQLite FTS5 over titles, answers, reasoning, previous attempts,
document text and attachment names. Space-separated terms are literal word prefixes,
not executable FTS expressions. Search ignores case and Latin accents. Results carry
message positions and jump to the matching turn; at most 200 hits are returned.
The normal sidebar shows the latest 500 conversations, pinned first.

`ChatStore::Export` writes a new folder with `chat.json`, `chat.md` and
`attachments/`. It also works for temporary chats and unsaved text after a save error.
JSON retains settings, model identity, timestamps and attempts. Missing media is
identified in Markdown; it is never silently replaced.
`ChatStore::Backup` uses SQLite's online backup API, checks the copied database,
and copies every referenced attachment after hash verification. An incomplete or
corrupt source fails the backup; only a successful bundle receives its final folder name.
Weights and app/server preferences are outside this conversation backup.

To restore, close Studio and preserve the existing data folder. Copy the backup's
`studio.db` and `attachments/` into an empty data folder (do not carry over old WAL/SHM
files), or launch Studio with `GEM16_STUDIO_DATA_ROOT` set to a copy of the backup.
No running database file should be overwritten. Schema migration builds the search
index transactionally; a newer schema remains protected from older Studio builds.

`ChatStore::CleanAttachments` removes only hash-named regular files without any
message reference, including archived chats. Profile/model blobs are never touched.
Backup bundles own their attachment copies and remain independent of this cleanup.
Tests additionally cover search escaping/accent handling, search migration, export,
backup reopening, FTS deletion and reference-aware garbage collection.
