# Abaddon Scout V1 Mission Brief

**Creator attribution:** Adrien D. Thomas

**Project:** GARVIS / Abaddon / Hypercube Heartbeat / ADOEG evidence workflow

**Prototype date:** 2026-08-05

## Mission

Abaddon Scout V1 is a bounded, read-only evidence examiner. It processes only
the evidence directory explicitly supplied by its operator.

The Scout records relative paths, byte sizes, SHA-256 digests, supplied
expectations, missing expected files, mismatches, and unresolved evidence. It
can produce `REPORT.json`, `manifest.sha256`, and a deterministic evidence ZIP.

## Containment rules

1. The Scout must not search outside the supplied evidence root.
2. Absolute paths, parent traversal, and symbolic links are rejected.
3. The output directory must remain outside the evidence root.
4. Importing the module must not execute a scan or create files.
5. The Scout has no network, account, email, GitHub, messaging, deletion,
   installation, purchasing, or protected-system capability.
6. Output files are created only by an explicit package-generation call.
7. Existing output files are never overwritten.
8. Evidence changes during packaging cause a fail-closed result.

## Claim boundaries

A matching SHA-256 digest verifies that examined bytes match a supplied digest.
It does not independently prove authorship, ownership, infringement, platform
conduct, scientific validity, historical custody, or account activity.

The Scout must preserve missing evidence, contradictions, and uncertainty
rather than infer unsupported conclusions.

## Prototype boundary

This four-file prototype may be tested only with temporary fixtures. It is not
authorized to inspect Adrien D. Thomas's real evidence package during this
stage.
