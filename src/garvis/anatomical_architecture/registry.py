"""Registry of the 11 anatomy-inspired GARVIS software systems."""

from __future__ import annotations

from .models import OrganSystem, SystemDefinition


SYSTEMS: dict[OrganSystem, SystemDefinition] = {
    OrganSystem.INTEGUMENTARY: SystemDefinition(
        system=OrganSystem.INTEGUMENTARY,
        biological_role="Protective outer barrier, temperature regulation, and fluid-loss prevention.",
        software_role="User-interface, API, authentication, permissions, and boundary-defense layer.",
        responsibilities=(
            "validate external input",
            "authenticate users and services",
            "enforce access boundaries",
            "sanitize display and output",
            "expose stable interfaces",
        ),
        inputs=("user input", "API requests", "files", "device signals"),
        outputs=("validated requests", "access decisions", "safe user output"),
        risks=("injection", "impersonation", "data exposure", "unsafe interface behavior"),
    ),
    OrganSystem.SKELETAL: SystemDefinition(
        system=OrganSystem.SKELETAL,
        biological_role="Structural support, protection, joints, and blood-cell production.",
        software_role="Core schemas, contracts, types, databases, module boundaries, and repository structure.",
        responsibilities=(
            "define stable data models",
            "provide architectural constraints",
            "protect critical state",
            "support module composition",
            "maintain versioned contracts",
        ),
        inputs=("requirements", "schemas", "configuration"),
        outputs=("types", "interfaces", "storage structures", "module boundaries"),
        risks=("schema drift", "tight coupling", "structural corruption", "breaking changes"),
    ),
    OrganSystem.MUSCULAR: SystemDefinition(
        system=OrganSystem.MUSCULAR,
        biological_role="Movement, circulation support, and transport through organs.",
        software_role="Executors, workers, tool drivers, job runners, and action adapters.",
        responsibilities=(
            "execute approved plans",
            "run background jobs",
            "operate tools",
            "move data through workflows",
            "report execution results",
        ),
        inputs=("action plans", "tool calls", "job instructions"),
        outputs=("executed work", "status", "errors", "measurements"),
        risks=("unapproved action", "runaway loops", "resource exhaustion", "partial failure"),
    ),
    OrganSystem.NERVOUS: SystemDefinition(
        system=OrganSystem.NERVOUS,
        biological_role="Sensory processing, coordination, control, and rapid response.",
        software_role="Central orchestration, neurocognitive heartbeat, event routing, recall, planning, and feedback.",
        responsibilities=(
            "interpret signals",
            "select relevant memory",
            "coordinate subsystems",
            "generate plans",
            "process feedback and errors",
        ),
        inputs=("sensory events", "memory", "system health", "goals"),
        outputs=("context", "plans", "routing decisions", "control signals"),
        risks=("context overload", "bad routing", "false certainty", "coordination failure"),
    ),
    OrganSystem.ENDOCRINE: SystemDefinition(
        system=OrganSystem.ENDOCRINE,
        biological_role="Slow hormonal regulation of metabolism, growth, stress, and reproduction.",
        software_role="Global policy, configuration, scheduling, rate limits, priorities, and long-timescale modulation.",
        responsibilities=(
            "set operating modes",
            "adjust priorities",
            "control schedules",
            "modulate resource use",
            "apply authority and safety policies",
        ),
        inputs=("policy", "operator settings", "time", "system state"),
        outputs=("configuration signals", "limits", "priorities", "scheduled cycles"),
        risks=("stale configuration", "conflicting policy", "over-throttling", "policy bypass"),
    ),
    OrganSystem.CARDIOVASCULAR: SystemDefinition(
        system=OrganSystem.CARDIOVASCULAR,
        biological_role="Transport of oxygen, nutrients, hormones, and waste.",
        software_role="Event bus, message queues, data transport, telemetry, and service-to-service communication.",
        responsibilities=(
            "transport events",
            "deliver state changes",
            "carry telemetry",
            "connect modules",
            "preserve message ordering where required",
        ),
        inputs=("events", "messages", "telemetry", "data packets"),
        outputs=("delivered signals", "queued work", "transport metrics"),
        risks=("message loss", "duplication", "backpressure", "latency", "dead letters"),
    ),
    OrganSystem.LYMPHATIC_IMMUNE: SystemDefinition(
        system=OrganSystem.LYMPHATIC_IMMUNE,
        biological_role="Pathogen defense, immune surveillance, and tissue-fluid return.",
        software_role="Security monitoring, anomaly detection, quarantine, integrity checks, and incident response.",
        responsibilities=(
            "detect suspicious behavior",
            "quarantine unsafe content",
            "verify integrity",
            "track incidents",
            "restore trusted state",
        ),
        inputs=("logs", "files", "network events", "tool results"),
        outputs=("risk scores", "quarantine decisions", "alerts", "recovery actions"),
        risks=("false positives", "missed attacks", "over-privilege", "contaminated evidence"),
    ),
    OrganSystem.RESPIRATORY: SystemDefinition(
        system=OrganSystem.RESPIRATORY,
        biological_role="Oxygen intake and carbon-dioxide removal.",
        software_role="Resource intake and exhaust: network I/O, compute budget, concurrency, voice airflow/prosody budget, and health checks.",
        responsibilities=(
            "manage input and output flow",
            "monitor compute and memory pressure",
            "control concurrency",
            "maintain service availability",
            "support voice timing and prosody",
        ),
        inputs=("network traffic", "compute demand", "audio streams"),
        outputs=("capacity signals", "throttle decisions", "health metrics"),
        risks=("overload", "timeouts", "thermal pressure", "network starvation"),
    ),
    OrganSystem.DIGESTIVE: SystemDefinition(
        system=OrganSystem.DIGESTIVE,
        biological_role="Breakdown of food, nutrient absorption, and solid-waste elimination.",
        software_role="Ingestion, parsing, normalization, extraction, chunking, indexing, and knowledge preparation.",
        responsibilities=(
            "parse documents and messages",
            "extract entities and claims",
            "normalize formats",
            "chunk large content",
            "prepare searchable knowledge",
        ),
        inputs=("documents", "email", "web pages", "audio transcripts", "code"),
        outputs=("normalized records", "chunks", "metadata", "candidate knowledge"),
        risks=("bad parsing", "lost context", "poisoned input", "duplicate knowledge"),
    ),
    OrganSystem.URINARY_EXCRETORY: SystemDefinition(
        system=OrganSystem.URINARY_EXCRETORY,
        biological_role="Blood filtration, metabolic-waste removal, and fluid-electrolyte balance.",
        software_role="Garbage collection, retention, deduplication, cache cleanup, log rotation, and context-pressure control.",
        responsibilities=(
            "remove redundant working data",
            "retain immutable archives",
            "rotate logs",
            "prune caches",
            "balance storage and context pressure",
        ),
        inputs=("temporary data", "logs", "duplicate records", "expired cache"),
        outputs=("clean working state", "retention reports", "archived records"),
        risks=("accidental deletion", "archive loss", "over-retention", "unbounded growth"),
    ),
    OrganSystem.REPRODUCTIVE: SystemDefinition(
        system=OrganSystem.REPRODUCTIVE,
        biological_role="Production of offspring and continuation of biological lineage.",
        software_role="Controlled generation of new modules, tests, templates, branches, agents, and reusable patterns.",
        responsibilities=(
            "scaffold new modules",
            "generate tests with code",
            "clone approved templates",
            "create experimental branches",
            "preserve lineage and authorship",
        ),
        inputs=("approved design", "templates", "requirements", "test criteria"),
        outputs=("new modules", "tests", "versioned branches", "lineage metadata"),
        risks=("uncontrolled replication", "unreviewed deployment", "license drift", "identity confusion"),
    ),
}


def get_system(system: OrganSystem | str) -> SystemDefinition:
    key = system if isinstance(system, OrganSystem) else OrganSystem(system)
    return SYSTEMS[key]


def list_systems() -> tuple[SystemDefinition, ...]:
    return tuple(SYSTEMS[system] for system in OrganSystem)
